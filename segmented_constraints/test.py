import json, pathlib, re, statistics, os
PART_HEADER_RE = re.compile(r"(?mi)^#part\s*(\d+)\s*$")
def slice_parts(content: str):
    headers = list(PART_HEADER_RE.finditer(content))
    if not headers:
        return {}
    parts = {}
    for j, hdr in enumerate(headers):
        n = int(hdr.group(1))
        s = hdr.end()
        e = headers[j + 1].start() if j + 1 < len(headers) else len(content)
        parts[n] = content[s:e].strip()
    return parts
input = '''#part 1
In the small town of Ravenswood, nestled in the English countryside, 25-year-old Emilia Grey struggled to come to terms with her past. Her life had been forever changed when her younger brother, Max, disappeared during a family vacation to the woods. The police had given up on the case years ago, but Emilia's determination to find him never wavered. She spent her days working as a librarian, surrounded by dusty books and forgotten stories, while her nights were filled with vivid dreams and restless searching. Emilia's obsession with finding Max often made her feel isolated and alone, but she couldn't shake the feeling that she was being watched, that someone was waiting for her to uncover the truth. As she locked up the library one evening, a mysterious letter slipped through the door, bearing a single sentence: "Look again at the place you least expect."

#part 2
The letter's cryptic message sent Emilia's mind racing. She couldn't help but think of the old, abandoned mine on the outskirts of town, a place she and Max had often explored as kids. Her best friend, Olivia, a local photographer, agreed to accompany her on a trip to the mine. As they made their way through the overgrown entrance, Emilia felt a shiver run down her spine. The air inside was thick with the scent of decay and neglect. Olivia, ever the pragmatist, urged caution, but Emilia's instincts told her that this was where she needed to start looking. They began to explore, their footsteps echoing off the walls as they stumbled upon an old journal belonging to a former miner. The entries spoke of strange occurrences and unexplained noises, but one passage caught Emilia's eye: "A boy went missing, and I fear I know who took him."

#part 3
As Emilia read the journal, a chill ran down her spine. The writer's words seemed to hint at a dark secret, one that had been hidden in plain sight. Suddenly, Olivia's voice cut through the silence, her tone low and urgent. "Emilia, look at this," she whispered, holding up a faded photograph. It was an old picture of the mine, but what caught Emilia's attention was the figure standing in the doorway – a figure that looked uncannily like her brother, Max. Emilia's heart skipped a beat as she realized that she might finally be closing in on the truth. But as she turned to leave, she heard the sound of footsteps, heavy and deliberate, coming from deeper within the mine. It was then that she knew she was not alone.

#part 4
The footsteps drew closer, and Emilia's heart pounded in her chest. She and Olivia turned to make their way back to the entrance, but it was too late. A figure emerged from the shadows, its features obscured by a hoodie. Emilia's instincts screamed at her to run, but her legs felt rooted to the spot. As the figure approached, it pushed back its hood, revealing a face that Emilia knew all too well – the face of her own father. The revelation hit her like a ton of bricks. Her father, the man she had trusted her whole life, was behind Max's disappearance. The truth spilled out, a mix of anger, sadness, and regret. In the end, Emilia found her brother, alive but traumatized, and her father was brought to justice. Though the journey had been long and arduous, Emilia finally found closure, but the scars of her ordeal would stay with her forever.

'''
parts = slice_parts(input)
for p, content in parts.items():
    print(p)
    print(content)
    print("*************************************************")