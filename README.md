# storyboard-ai

## AI Generated: The Story of Shepherd Boy And The Wolf

> **I am still under the process of cleaning and re-structuring the codebase**

---

### Turning this into a product

Work is underway to turn this research pipeline into a SaaS for solo content creators - story text
in, narrated whiteboard video out.

- [`docs/product-plan.md`](docs/product-plan.md) - product definition, decisions, phased roadmap
- [`docs/architecture.md`](docs/architecture.md) - system design and the parity constraints
- [`docs/cost-model.md`](docs/cost-model.md) - per-video COGS and pricing analysis
- [`prototype/README.md`](prototype/README.md) - Remotion port, **pixel-identical** to the
  renderer below, plus how to reproduce that result

---

https://www.youtube.com/watch?v=iSb1HJXRO04  
[![https://www.youtube.com/watch?v=iSb1HJXRO04](https://img.youtube.com/vi/iSb1HJXRO04/0.jpg)](https://www.youtube.com/watch?v=iSb1HJXRO04)


### HOW THE VIDEO WAS MADE ? 

- The story was taken from internet.
- A 4 line summary of the story was generated using ChatGPT
- For each summary line, a corresponding image was generated using Stable Diffusion
- In each image (4 in our case), the important object were masked using MetaAI SAM 
- I have developed a custom image to white-board animation code which converted the images to videos (This was the most time-consuming part of the development process). 
- The audio was generated using gTTS (Google Text-to-Speech)
- The sub-titles were generated with the help of OpenAI whisper 
- All the different audios, videos and subtitles were somehow synchronized using FFMPEG and OpenCV (This part gave a lot of pain🥲... ALL HAIL FFMPEG🙌)


---

### Code Structure: 

- **text-processing**: 
    - **story.txt**: The story has been taken from internet
    - **summary.txt**: The 4 line summary was generated using chatGPT (for the story present in story.txt)
- **generate-whiteboard-animated-videos**: 
    - Convert any image to white-board animations video
    - If object masks are provided in JSON formats, they will be automatically considered while drawing 
    - **Code File:** draw-whiteboard-animations.py
- **tts (text to speech)**:
    - The story is devided into 4 parts in correspondance to the summary. For each story part, google's text-2-speec used to generate 4 mp3 files. 

---

## Sam assisted mask generation:

- The whiteboard animation is refined by drawing one object at a time. 
- These object masks are drawn/created using **Meta-AI SAM** assisted tagging tool named: **Anylabeling** 
- **Anylabeling Tool**: https://github.com/vietanhdev/anylabeling 
- Post tagging, the tool returns a JSON output which can be used for sketching video generation 

```
Please NOTE:
- The videos can also be generated using without the JSON masks 
```