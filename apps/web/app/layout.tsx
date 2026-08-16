import type { Metadata } from "next";

import "./globals.css";
import { account } from "@/lib/demoData";

export const metadata: Metadata = {
  title: "storyboard-ai",
  description: "Turn a story into a narrated whiteboard animation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const used = Math.min(account.rendersUsed / account.quota, 1);

  return (
    <html lang="en">
      <body>
        <header className="top">
          <div className="inner">
            <div className="logo">
              story<span>board</span>
            </div>
            <nav>
              <a href="/">Projects</a>
              <a href="/pricing">Pricing</a>
            </nav>
            <div className="spacer" />
            <div className="quota">
              <span>
                {account.rendersUsed} of {account.quota} renders this {account.quotaPeriod}
              </span>
              <span className="bar">
                <i style={{ width: `${used * 100}%` }} />
              </span>
              <span>{account.plan}</span>
            </div>
          </div>
        </header>
        <main className="shell">{children}</main>
      </body>
    </html>
  );
}
