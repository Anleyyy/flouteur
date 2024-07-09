import sys
from cx_Freeze import setup, Executable


# base="Win32GUI" should be used only for Windows GUI app
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name = "Flouteur",
    version = "hs",
    description = "Une application qui floute les visages présent sur une vidéo.",
    options = {"build_exe": {"include_files":["buy_moree.ico","ffmpeg-4.4-full_build","icon","model","protos","utils"]}},
    executables = [Executable("main61s.py", base=base)]
)
