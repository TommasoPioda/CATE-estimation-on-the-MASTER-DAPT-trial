-- Pandoc filter: map the few characters the text fonts do not carry.
-- Emoji (U+2705, U+26A0 + variation selector) would render as tofu in XeLaTeX
-- with DejaVu; the plain dingbat equivalents are present and mean the same thing.
local subs = {
  ["\u{FE0F}"] = "",         -- variation selector-16 (forces emoji presentation)
  ["\u{2705}"] = "\u{2714}", -- white heavy check mark -> heavy check mark
}

function Str(el)
  for from, to in pairs(subs) do
    el.text = el.text:gsub(from, to)
  end
  return el
end
