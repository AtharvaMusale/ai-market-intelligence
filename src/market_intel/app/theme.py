"""Visual theme matching atharvamusale.github.io: black base, hairline rails with crosshairs,
Geist / Geist Mono, mono uppercase labels, spectrum-gradient accent, white pill buttons."""
from __future__ import annotations

import streamlit as st

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');
:root{--bg:#000;--bg-2:#0a0a0a;--bg-3:#111;--text:#ededed;--muted:#a1a1a1;--faint:#8a8a8a;--line:#1f1f1f;--line-2:#2e2e2e;
--plus:#5a5a5a;--link:#52a8ff;--green:#45c26b;--amber:#f5a524;--rose:#f43f5e;--violet:#b675f1;--grid:rgb(255 255 255/.06);
--spectrum:linear-gradient(90deg,#22d3ee,#a855f7 28%,#f43f5e 52%,#f59e0b 76%,#10b981);
--mono:'Geist Mono',ui-monospace,SFMono-Regular,Menlo,monospace;--pad:40px}
html,body,.stApp,.stApp *:not(code):not(.material-symbols-rounded):not([data-testid="stIconMaterial"]){font-family:'Geist',ui-sans-serif,system-ui,-apple-system,sans-serif}
.stApp{background:var(--bg);color:var(--text);-webkit-font-smoothing:antialiased}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:var(--bg-2);border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{font:500 12px/1 var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}
.block-container{max-width:1200px;padding:0 var(--pad) 40px!important;border-inline:1px solid var(--line)}
a{color:var(--link)} a:hover{text-decoration:underline;text-underline-offset:3px}
::selection{background:rgb(182 117 241/.4)}

/* nav */
.nav{display:flex;align-items:center;justify-content:space-between;height:64px;margin:0 calc(-1*var(--pad));padding:0 var(--pad);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em;color:var(--text)}
.brand-mark{width:22px;height:22px;background:var(--text);clip-path:polygon(50% 8%,100% 92%,0 92%)}
.nav-note{font:400 12px/1 var(--mono);color:var(--faint);text-transform:uppercase;letter-spacing:.06em;margin-right:56px}

/* hero */
.hero{position:relative;text-align:center;margin:0 calc(-1*var(--pad));padding:76px 24px 68px;
background:radial-gradient(55% 75% at 12% 105%,rgb(59 130 246/.24),transparent 70%),radial-gradient(45% 70% at 50% 110%,rgb(244 63 94/.18),transparent 70%),radial-gradient(50% 75% at 90% 100%,rgb(245 158 11/.16),transparent 70%),
linear-gradient(var(--grid) 1px,transparent 1px) 0 0/80px 80px,linear-gradient(90deg,var(--grid) 1px,transparent 1px) 0 0/80px 80px}
.pill{display:inline-flex;align-items:center;gap:9px;border:1px solid var(--line-2);background:var(--bg-2);color:var(--muted);border-radius:999px;padding:8px 16px;font-size:14px;margin-bottom:26px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 3px rgb(69 194 107/.22)}
.hero .h1{font-size:clamp(36px,5.4vw,64px);font-weight:600;letter-spacing:-.035em;line-height:1.05;max-width:880px;margin:0 auto 22px}
.hero p{color:var(--muted);font-size:18px;line-height:1.6;max-width:700px;margin:0 auto}
.spectrum{background:var(--spectrum);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shift 8s ease-in-out infinite alternate}
@keyframes shift{from{background-position:0 50%}to{background-position:100% 50%}}

/* rails + crosshairs */
.rule{position:relative;height:1px;background:var(--line);margin:0 calc(-1*var(--pad))}
.rule::before,.rule::after{content:"";position:absolute;top:-7px;width:15px;height:15px;
background:linear-gradient(var(--plus),var(--plus)) center/1px 100% no-repeat,linear-gradient(var(--plus),var(--plus)) center/100% 1px no-repeat}
.rule::before{left:-1px;transform:translateX(0)}.rule::after{right:-1px}
.spacer{height:44px}

/* tiles */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));border-top:1px solid var(--line);margin:0 calc(-1*var(--pad))}
.tile{padding:26px 28px 24px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.tile:last-child{border-right:0}
.tile .v{font-size:32px;font-weight:600;letter-spacing:-.025em;line-height:1.1}
.tile .k{color:var(--muted);font-size:14px;margin-top:8px}
.tile .s{font:400 12px/1.4 var(--mono);color:var(--faint);text-transform:uppercase;letter-spacing:.05em;margin-top:12px}

@media(max-width:900px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))!important}.tile:nth-child(2n){border-right:0}}
/* section headings */
.label{display:inline-flex;align-items:center;gap:10px;margin:0 0 16px;font:500 12px/1 var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}
.label span:first-child{color:var(--text);padding:4px 7px;border:1px solid var(--line-2);border-radius:5px;background:var(--bg-2)}
.h2{font-size:34px;font-weight:600;letter-spacing:-.03em;line-height:1.1;margin:0 0 10px}
.sub{color:var(--muted);font-size:17px;line-height:1.55;max-width:680px;margin:0 0 26px}

/* claims */
.claims{list-style:none;padding:0!important;margin:0 0 34px!important;border-top:1px solid var(--line)}
.claims li{padding:18px 0!important;margin:0!important;border-bottom:1px solid var(--line)}
.claims p{margin:0 0 10px;font-size:17px;line-height:1.5;color:var(--text)}
.chip{display:inline-block;font:400 12px/1 var(--mono);color:var(--muted);background:var(--bg-2);border:1px solid var(--line);padding:6px 8px;border-radius:6px;margin:0 6px 6px 0}
a.chip{color:var(--link)} a.chip:hover{text-decoration:none;border-color:var(--plus)}
.sec-title{font:500 12px/1 var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin:30px 0 14px;display:flex;gap:10px;align-items:center}
.sec-title span:first-child{color:var(--text);padding:4px 7px;border:1px solid var(--line-2);border-radius:5px;background:var(--bg-2)}
.empty{color:var(--muted);border:1px dashed var(--line-2);border-radius:12px;padding:28px;text-align:center}
.ok{color:var(--green)} .bad{color:var(--rose)}

/* widgets */
button[data-testid="stBaseButton-primary"]{background:var(--text);color:#000;border:1px solid var(--text);border-radius:999px;height:44px;padding:0 24px;font-weight:500;transition:background .15s}
button[data-testid="stBaseButton-primary"] p{color:#000!important;font-size:15px}
button[data-testid="stBaseButton-primary"]:hover{background:#d0d0d0;border-color:#d0d0d0}
button[data-testid="stBaseButton-secondary"]{background:var(--bg-2);color:var(--text);border:1px solid var(--line-2);border-radius:999px}
[role="tablist"]{gap:4px;border-bottom:1px solid var(--line)}
[role="tab"]{background:transparent;color:var(--muted);border-radius:999px;padding:8px 16px;height:auto;margin-bottom:10px;transition:color .15s,background .15s}
[role="tab"] p{color:inherit;font-size:14px}
[role="tab"][aria-selected="true"]{background:var(--bg-3);color:var(--text)}
[role="tab"]:hover{color:var(--text)}
[class*="SelectionIndicator"],[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}
div[data-baseweb="input"],div[data-baseweb="base-input"]{background:var(--bg-2)!important;border-color:var(--line-2)!important;border-radius:10px!important}
[data-testid="stExpander"] details{border:1px solid var(--line);border-radius:10px;background:var(--bg-2)}
[data-testid="stAlert"]{background:var(--bg-2);border:1px solid var(--line-2);border-radius:10px}
[data-testid="stCaptionContainer"]{color:var(--faint)}
[data-testid="stCheckbox"] p,[data-testid="stRadio"] p{color:var(--muted)}

/* footer */
.footer{display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;margin:60px calc(-1*var(--pad)) 0;padding:24px var(--pad) 0;border-top:1px solid var(--line);color:var(--faint);font-size:14px}
.footer a{color:var(--muted)}
"""


def inject() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
