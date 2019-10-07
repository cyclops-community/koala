"""
This module checks if CTF Python is installed.
"""

try:
    import ctf
    found_ctf = True
except:
    found_ctf = False
