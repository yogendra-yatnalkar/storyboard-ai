-- storyboard-ai initial schema
--
-- Row-level security is on for every table and the policies are written so a
-- user can only ever reach their own rows. This is not optional detail: the
-- Supabase anon key ships to the browser, so RLS is the actual access control,
-- not a second layer behind an API that enforces it.
--
-- Render jobs are deliberately a separate table from projects. A project is
-- user-editable state; a job is an audit trail of what was actually spent and
-- produced, and it must survive the project being edited or deleted.

create extension if not exists "pgcrypto";

-- --- accounts ---------------------------------------------------------------

create type plan_id as enum ('free', 'pro', 'studio');
create type subscription_status as enum ('active', 'past_due', 'cancelled', 'none');

create table accounts (
  id uuid primary key references auth.users (id) on delete cascade,
  email text not null,
  email_verified boolean not null default false,
  plan_id plan_id not null default 'free',
  status subscription_status not null default 'none',
  -- Videos rendered in the current period. Free plans use a lifetime quota, so
  -- this counter simply never resets for them.
  renders_used integer not null default 0,
  period_start timestamptz,
  -- Merchant-of-record subscription reference (Paddle / Lemon Squeezy).
  provider_customer_id text,
  provider_subscription_id text,
  -- Guards against out-of-order webhook delivery overwriting newer state.
  billing_updated_at timestamptz,
  created_at timestamptz not null default now()
);

alter table accounts enable row level security;

create policy "accounts are self-readable"
  on accounts for select
  using (auth.uid() = id);

-- Deliberately no insert/update policy: plan, status and usage are written only
-- by the service role from webhook and render paths. A user who could update
-- their own row could grant themselves Studio.

-- --- projects ---------------------------------------------------------------

create type aspect_ratio as enum ('square', 'landscape', 'portrait');
create type caption_style as enum ('word', 'sentence', 'none');

create table projects (
  id uuid primary key default gen_random_uuid(),
  account_id uuid not null references accounts (id) on delete cascade,
  title text not null default 'Untitled',
  -- The story text the creator typed.
  input text not null default '',
  aspect aspect_ratio not null default 'portrait',
  caption_style caption_style not null default 'word',
  language text not null default 'en',
  segmentation boolean not null default true,
  -- Storyboard as returned by the LLM, plus any user edits.
  storyboard jsonb not null default '{"scenes": []}'::jsonb,
  brand jsonb not null default '{}'::jsonb,
  music text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table projects enable row level security;

create policy "projects are owner-managed"
  on projects for all
  using (auth.uid() = account_id)
  with check (auth.uid() = account_id);

create index projects_account_updated_idx on projects (account_id, updated_at desc);

-- --- render jobs ------------------------------------------------------------

create type job_status as enum ('queued', 'generating', 'rendering', 'complete', 'failed');

create table render_jobs (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references projects (id) on delete cascade,
  account_id uuid not null references accounts (id) on delete cascade,
  status job_status not null default 'queued',
  progress real not null default 0,
  error text,
  -- Constraints resolved from the plan at enqueue time, so a mid-render plan
  -- change cannot retroactively remove a watermark.
  watermark boolean not null default true,
  height integer not null default 720,
  tier text not null default 'cheap',
  -- Where the finished video landed (R2 key, not a signed URL - those expire).
  output_key text,
  duration_seconds real,
  -- Measured, not estimated: what this video actually cost to produce.
  cost_usd numeric(10, 5),
  usage jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  completed_at timestamptz
);

alter table render_jobs enable row level security;

create policy "jobs are owner-readable"
  on render_jobs for select
  using (auth.uid() = account_id);

-- Jobs are created and advanced by the service role only; a user-writable job
-- row would let someone set status='complete' and skip the quota decrement.

create index render_jobs_project_idx on render_jobs (project_id, created_at desc);
create index render_jobs_status_idx on render_jobs (status) where status in ('queued', 'generating', 'rendering');

-- --- webhook idempotency ----------------------------------------------------

-- Merchant-of-record providers retry, so the same event arrives more than once.
create table processed_webhook_events (
  id text primary key,
  received_at timestamptz not null default now()
);

alter table processed_webhook_events enable row level security;
-- No policies: service role only.

-- --- quota ------------------------------------------------------------------

-- Claiming a render slot atomically. Doing this as a read-then-write in
-- application code races: two concurrent renders both read renders_used = 29
-- against a quota of 30 and both proceed.
create or replace function claim_render_slot(p_account_id uuid, p_quota integer)
returns boolean
language plpgsql
security definer
as $$
declare
  claimed boolean;
begin
  update accounts
     set renders_used = renders_used + 1
   where id = p_account_id
     and renders_used < p_quota
  returning true into claimed;

  return coalesce(claimed, false);
end;
$$;

-- --- housekeeping -----------------------------------------------------------

create or replace function touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger projects_touch_updated_at
  before update on projects
  for each row execute function touch_updated_at();

-- New auth users get an account row automatically, so the app never has to
-- handle a signed-in user with no account.
create or replace function handle_new_user()
returns trigger
language plpgsql
security definer
as $$
begin
  insert into accounts (id, email, email_verified)
  values (new.id, new.email, new.email_confirmed_at is not null)
  on conflict (id) do nothing;
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function handle_new_user();
