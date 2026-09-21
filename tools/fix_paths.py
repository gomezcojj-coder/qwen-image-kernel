# Replace hardcoded personal absolute paths in tools/tests with a portable ROOT shim.
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = re.compile(r'sys\.path\.insert\(0, r?[\'"]C:\\Users\\gomez\\Desktop\\Coding_Projects\\Qwen_Image[\'"]\)')

shim = (
    "from pathlib import Path\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
    "import os as _os\n"
    "_os.chdir(Path(__file__).resolve().parent.parent)"
)

changed = 0
for py in list(Path("tools").glob("*.py")) + list(Path("tests").glob("*.py")) + list(Path(".").glob("*.py")):
    src = py.read_text(encoding="utf-8", errors="ignore")
    if "C:\\Users\\gomez" not in src and "C:/Users/gomez" not in src:
        continue
    new = OLD.sub(shim, src)
    # catch any remaining absolute-path literals (single vs double quote variants)
    new = new.replace('r"C:\\Users\\gomez\\Desktop\\Coding_Projects\\Qwen_Image"',
                      'str(Path(__file__).resolve().parent.parent)')
    new = new.replace("r'C:\\Users\\gomez\\Desktop\\Coding_Projects\\Qwen_Image'",
                      "str(Path(__file__).resolve().parent.parent)")
    py.write_text(new, encoding="utf-8")
    changed += 1
    print("fixed", py)
print("changed:", changed)