"""Set the version number in one of the files that declare it.

Usage: python update_version.py <file> <version>

Each pattern below targets exactly one declaration style. A file that matches
none of them is treated as an error rather than rewritten unchanged, so a
renamed key or a reformatted file fails the release run instead of shipping a
stale version number.
"""
import re
import sys

# (pattern, flags) -- group 1 is the prefix up to the opening quote, group 2 the
# closing quote, so the version itself is whatever sits between them.
PATTERNS = (
    # pyproject.toml       version = "1.19"
    (r'^(version\s*=\s*")[^"]*(")', re.M),
    # docs/source/conf.py  release = "1.19"
    (r'^(release\s*=\s*")[^"]*(")', re.M),
    # conda-recipe/meta.yaml  {% set version = '1.19' %}
    (r"(\{%\s*set\s+version\s*=\s*')[^']*(')", 0),
)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    path, version = sys.argv[1], sys.argv[2]

    with open(path, 'r') as f:
        content = f.read()

    total = 0
    for pattern, flags in PATTERNS:
        content, count = re.subn(pattern, r'\g<1>' + version + r'\g<2>', content, flags=flags)
        total += count

    if not total:
        sys.exit(f'{path}: no version declaration matched, version NOT updated')

    with open(path, 'w') as f:
        f.write(content)
    print(f'{path}: set version to {version} ({total} occurrence(s))')


if __name__ == '__main__':
    main()
