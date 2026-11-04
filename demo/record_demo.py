"""Record the synthetic-only demo: one screen per synthetic report, then a closing card -> demo/CMR_scar_demo_synthetic.mp4.

Start the app first (CMR_DEMO_NO_BROWSER=1 python demo/app.py), then run this with Playwright and Pillow installed
(pip install playwright pillow; python -m playwright install chromium). Needs ffmpeg. Frames go to a temporary folder."""
import os, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

SP = tempfile.mkdtemp(); FR = os.path.join(SP, "frames"); os.makedirs(FR)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CMR_scar_demo_synthetic.mp4")
W, H = 1920, 1200
CASES = ["Ischemic", "Non-ischemic", "Mixed", "No-scar", "Not-specified", "No scar, negated scar words"]

def card(path, title, lines):
    im = Image.new("RGB", (W, H), "#5E1220"); d = ImageDraw.Draw(im)
    f1 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 64)
    f2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 38)
    d.text((140, 380), title, font=f1, fill="white")
    for i, l in enumerate(lines): d.text((140, 520 + i * 70), l, font=f2, fill="#F3E3E6")
    im.save(path)

# No opening title card: the video starts on the app (author request).

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.2, color_scheme="light")
    pg.goto(os.environ.get("CMR_DEMO_URL", "http://127.0.0.1:7860")); pg.wait_for_timeout(3000)
    names = []
    for i, name in enumerate(CASES, 1):
        pg.get_by_label("Synthetic report").first.click(); pg.wait_for_timeout(400)
        pg.get_by_role("option", name=name, exact=True).click(); pg.wait_for_timeout(1500)
        # wait for the LLM card to finish: an answer, an agreement line, or "NLP only"
        pg.wait_for_function("""() => { const t = document.body.innerText;
            return !t.includes('Sending the report') && (t.includes('Agrees with the rules') || t.includes('Differs from the rules') || t.includes('NLP only')); }""", timeout=180000)
        pg.wait_for_timeout(800)
        f = os.path.join(FR, f"{i:02d}.png"); pg.screenshot(path=f); names.append(f)
    b.close()

card(os.path.join(FR, "99.png"), "Rules and LLMs together",
     ["Zero-shot: the paper's prompt, no labeled examples.", "Few-shot: 5 expert-labeled examples per report, chosen by similarity.",
      "Agree with the rules: accept. Disagree: send for expert review.", os.environ.get("CMR_DEMO_SUMMARY", "")])

frames = sorted(os.path.join(FR, f) for f in os.listdir(FR) if f.endswith(".png"))
lst = os.path.join(SP, "frames.txt")
with open(lst, "w") as fh:
    for f in frames:
        dur = 4 if os.path.basename(f) == "99.png" else 6
        fh.write(f"file '{f}'\nduration {dur}\n")
    fh.write(f"file '{frames[-1]}'\n")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,fps=24,format=yuv420p",
                "-c:v", "libx264", "-crf", "20", OUT], check=True)
print("wrote", OUT, len(frames), "frames")
