# Local Offline Visual Acceptance

Use this checklist for the local-model offline karaoke flow before any formal tester sign-off.

## Required evidence

- Tester must capture their own keyframes from the current run's final MP4.
- For each focus point, capture:
  - one full-frame image
  - one subtitle-band crop
  - a 200%-300% zoom crop when ruby/furigana placement is ambiguous
  - a compare board when comparing against a prior rejected run
- Do not accept based on ASS files, logs, or generated metadata alone.

## Output contract

- Default preview, default primary output, default primary download, and default delivered MP4 must preserve vocals.
- Instrumental or no-vocals video may exist only as a secondary output.
- Final accepted MP4 must remain 1280x720.
- Canonical lyrics remain the only allowed lyric body source.
- Final MP4 must not contain text beyond lyric body and necessary ruby/furigana.
- Highlight progression must move left-to-right, not jump entire lines.

## Visual focus points

- Focus 0: default vocals + WebUI primary output + opening/closing no-extra-text
- Focus A: あ、叫びたい伝えたい君に、今 / もっと知りたくて、もっと見たくて
- Focus B: 君とずっと会いたい / 好きな気持ちで
- Focus C: 君と一緒ならば / 広がるパノラマ
- Focus D: 私のことを全然 / 気にしないみたいけど
- Focus E: 一気に心を伝えるよ!、今 / もっと会いたくて、もっと語りたい
- Focus F: 君だけに恋をあげたい / 重なる未来を向いて
- Focus G: それだけで良い、だから / この瞬間いつまでも守りたい
- Focus H: every focus above must have full-frame + subtitle-band crop evidence
