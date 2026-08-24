export default function NewProjectPage() {
  return (
    <>
      <h1>New video</h1>
      <p className="lede">
        Paste a story or describe a topic. Scenes, illustrations, narration and captions are
        generated for you.
      </p>

      <div className="panel">
        <label htmlFor="story">Your story</label>
        <textarea
          id="story"
          defaultValue={
            "Once, there was a boy who watched over the village sheep on a hillside. Bored, he cried \"Wolf!\" and the villagers came running..."
          }
        />

        <div className="fields">
          <div>
            <label htmlFor="aspect">Aspect ratio</label>
            <select id="aspect" defaultValue="portrait">
              <option value="portrait">9:16 — Shorts, Reels, TikTok</option>
              <option value="landscape">16:9 — YouTube</option>
            </select>
          </div>
          <div>
            <label htmlFor="scenes">Scenes</label>
            <select id="scenes" defaultValue="4">
              <option value="3">3 scenes</option>
              <option value="4">4 scenes</option>
              <option value="5">5 scenes</option>
              <option value="6">6 scenes</option>
            </select>
          </div>
          <div>
            <label htmlFor="voice">Voice</label>
            <select id="voice" defaultValue="warm">
              <option value="warm">Warm narrator</option>
              <option value="bright">Bright and quick</option>
              <option value="calm">Calm explainer</option>
            </select>
          </div>
        </div>

        <div className="row" style={{ marginTop: 22 }}>
          <span className="note" style={{ margin: 0 }}>
            Estimated cost to you: 1 render from your monthly quota.
          </span>
          <button className="button" type="button">
            Generate storyboard
          </button>
        </div>
      </div>

      <p className="note">
        The storyboard is free to preview and regenerate — a render is only spent when you export
        the video.
      </p>
    </>
  );
}
