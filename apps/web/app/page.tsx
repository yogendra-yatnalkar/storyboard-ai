import { projects, type DemoProject } from "@/lib/demoData";

const ASPECT_LABEL = {
  portrait: "9:16",
  landscape: "16:9",
  square: "1:1",
} as const;

function StatusBadge({ status }: { status: DemoProject["status"] }) {
  if (status === "complete") return <span className="badge complete">Ready</span>;
  if (status === "failed") return <span className="badge">Failed</span>;
  const label = status === "queued" ? "Queued" : status === "generating" ? "Generating" : "Rendering";
  return <span className="badge busy">{label}</span>;
}

export default function ProjectsPage() {
  return (
    <>
      <div className="row">
        <div>
          <h1>Projects</h1>
          <p className="lede">Paste a story, get a narrated whiteboard video.</p>
        </div>
        <a className="button" href="/projects/new">
          New video
        </a>
      </div>

      <div className="grid">
        {projects.map((project) => (
          <a className="card" key={project.id} href={`/projects/${project.id}`}>
            <div className="thumb" style={{ background: project.thumbnail }}>
              {ASPECT_LABEL[project.aspect]}
            </div>
            <div className="body">
              <h3>{project.title}</h3>
              <div className="meta">
                <StatusBadge status={project.status} />
                <span>{project.sceneCount} scenes</span>
                {project.durationSeconds > 0 ? (
                  <span>{project.durationSeconds.toFixed(0)}s</span>
                ) : null}
              </div>
              {project.status !== "complete" && project.status !== "failed" ? (
                <div className="progress">
                  <i style={{ width: `${project.progress * 100}%` }} />
                </div>
              ) : null}
              <div className="meta" style={{ marginTop: 10 }}>
                <span>{project.updatedAt}</span>
                {project.costUsd !== null ? <span>${project.costUsd.toFixed(2)}</span> : null}
              </div>
            </div>
          </a>
        ))}
      </div>
    </>
  );
}
