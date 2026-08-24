import { costBreakdown, projects, shepherdScenes } from "@/lib/demoData";

export default async function ProjectPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const project = projects.find((p) => p.id === id) ?? projects[0];
  const total = costBreakdown.reduce((sum, row) => sum + row.value, 0);

  return (
    <>
      <div className="row">
        <div>
          <h1>{project.title}</h1>
          <p className="lede">
            {project.sceneCount} scenes &middot; {project.durationSeconds.toFixed(1)}s &middot;{" "}
            {project.aspect === "portrait" ? "9:16 vertical" : "16:9 landscape"} &middot; word captions
          </p>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <a className="button accent" href="#">
            Download MP4
          </a>
          <a className="button" href="#">
            Re-render
          </a>
        </div>
      </div>

      <h2>Storyboard</h2>
      <div className="panel">
        {shepherdScenes.map((scene) => (
          <div className="scene" key={scene.index}>
            <div className="n">{scene.index}</div>
            <div>
              <p className="narration">{scene.narration}</p>
              <p className="prompt">{scene.imagePrompt}</p>
            </div>
            <div className="timing">
              {scene.durationSeconds.toFixed(2)}s
              <br />
              {scene.captionWords} words
            </div>
          </div>
        ))}
      </div>
      <p className="note">
        Drawing speed is solved against each scene&rsquo;s narration, so the hand finishes as the
        voice does — every scene lands within one frame of its audio.
      </p>

      <h2>What this video cost</h2>
      <div className="panel">
        <table>
          <thead>
            <tr>
              <th>Item</th>
              <th className="num">USD</th>
            </tr>
          </thead>
          <tbody>
            {costBreakdown.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td className="num">{row.value.toFixed(4)}</td>
              </tr>
            ))}
            <tr>
              <td>
                <strong>Total</strong>
              </td>
              <td className="num">
                <strong>{total.toFixed(4)}</strong>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="note">
        Measured from the actual provider calls rather than estimated, so plan quotas can be set
        from data.
      </p>
    </>
  );
}
