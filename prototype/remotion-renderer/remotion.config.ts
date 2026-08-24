import { Config } from "@remotion/cli/config";

// PNG, not JPEG. Remotion captures each frame as a still before encoding, and
// JPEG's ringing around black strokes on white paper costs ~8/255 mean error per
// frame - enough to swamp a parity comparison against the reference renderer.
Config.setVideoImageFormat("png");
Config.setEntryPoint("./src/index.ts");

// The drawing is line art on white: high quality keeps the thin strokes from
// smearing, which would make a visual comparison against the reference useless.
Config.setCrf(16);

// A container-friendly Chromium is preinstalled in this environment; Remotion
// otherwise downloads its own headless shell on first render.
if (process.env.REMOTION_BROWSER_EXECUTABLE) {
  Config.setBrowserExecutable(process.env.REMOTION_BROWSER_EXECUTABLE);
}

export {};
