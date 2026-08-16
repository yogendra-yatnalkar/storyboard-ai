# Screenshots

All captured from the running code, not mocked up.

## Renderer output

Frames rendered by Remotion from the four-scene Shepherd Boy fixture.

| File | What it shows |
|---|---|
| `01-square-sentence-captions.png` | 1:1 render with phrase-block captions |
| `02-landscape-sentence-captions.png` | 16:9, square art padded rather than stretched |
| `03-portrait-word-captions.png` | 9:16 with word-by-word highlighting, mid-drawing with the hand visible |
| `04-portrait-scene3.png` | A later scene, showing the object-by-object colour reveal |
| `05-parity-target-frame385.png` | Frame 385 of the parity composition — pixel-identical to the Python renderer |

Reproduce with:

```bash
cd prototype/remotion-renderer
export REMOTION_BROWSER_EXECUTABLE=/path/to/chrome-headless-shell
npx remotion still Story-Portrait out.png --frame=700
```

## Web app

Captured with Playwright against `npm start` at 2x device scale.

| File | Page |
|---|---|
| `10-app-projects.png` | Projects list with render status and per-video cost |
| `11-app-new-video.png` | New video form |
| `12-app-project-detail.png` | Storyboard and measured cost breakdown |
| `13-app-pricing.png` | Plans, at the revised quotas from the cost model |

The app renders from `apps/web/lib/demoData.ts`, so these show real UI with
stand-in data — it is not yet connected to a database. See
[../status.md](../status.md).
