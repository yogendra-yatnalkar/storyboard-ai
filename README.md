# storyboard-ai

## AI Generated: The Story of Shepherd Boy And The Wolf

> **This code is under active development. Sorry, I am still under the process of cleaning and re-structuring the codebase**

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
