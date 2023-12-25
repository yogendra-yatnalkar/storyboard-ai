from gtts import gTTS
import os

save_folder = "./saved_audios"

story = [
    "Once, there was a boy who became bored when he watched over the village sheep grazing on the hillside. To entertain himself, he sang out, \"Wolf! Wolf! The wolf is chasing the sheep! When the villagers heard the cry, they ran up the hill to drive the wolf away.\"",
    "But when they arrived, they saw no wolf. The boy was amused when he saw their angry faces. \"Don\'t scream wolf when there is no wolf, boy!\" the villagers warned. They angrily went back down the hill. Later, the shepherd boy cried out once again, \"Wolf! Wolf! The wolf is chasing the sheep!\" To his amusement, the villagers came running up the hill to scare the wolf away. As they saw there was no wolf, they said strictly, \"Save your frightened cry for when there really is a wolf! Don\'t cry \'wolf\' when there is no wolf!\" But the boy grinned at their words while they walked, grumbling down the hill once more.",
    "The last time, a real wolf came and wanted to eat his sheep. The boy said help, help, but the people did not listen to him. They thought he was lying again. They did not help him and he lost his sheep.",
    "The story ends with an old man telling the boy that lying is bad and people will not trust you if you lie. The Moral of the story is - Lying breaks trust, and no one believes a liar — even when they tell the truth.",
]


for i in range(len(story)):
    tts = gTTS(story[i])
    tts.save(os.path.join(save_folder, f'{i}.mp3'))