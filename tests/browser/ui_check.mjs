// Drives the real UI mid-run: tab switching must not freeze the page (Gradio's Svelte update loop), the loss chart
// draws and keeps updating, previews show, Network Alpha exists, settings come back after a reload, and choosing
// MOSS-Image swaps in its own controls (and back), a choice that also survives a reload.
// usage: node ui_check.mjs <app url> <screenshot.png>   (prints PASS/FAIL lines; exit code 1 on failure)
// Browser: CHROME_PATH (e.g. C:\Program Files\Google\Chrome\Application\chrome.exe or msedge.exe), else @sparticuz/chromium.
import puppeteer from "puppeteer-core";
const [url, shot] = process.argv.slice(2);
let executablePath = process.env.CHROME_PATH, args = ["--no-sandbox"];
if (!executablePath) {
  const chromium = (await import("@sparticuz/chromium")).default;
  executablePath = await chromium.executablePath(); args = chromium.args;
}
let failed = 0;
const check = (name, ok, detail = "") => { if (!ok) failed++; console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail !== "" ? `  [${detail}]` : ""}`); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const browser = await puppeteer.launch({ executablePath, args, headless: true, defaultViewport: { width: 1600, height: 1000 } });
try {
  const page = await browser.newPage(); let pageErrors = 0;
  page.on("pageerror", () => pageErrors++);
  await page.goto(url, { waitUntil: "networkidle2" }); await sleep(3000);
  const tab = async name => { const [b] = await page.$$(`xpath/.//button[@role="tab" and normalize-space(.)="${name}"]`); await b.click(); await sleep(2500); };
  const alive = async () => Promise.race([page.evaluate(() => true), sleep(4000).then(() => false)]);
  const step = async () => Number((await page.evaluate(() => document.body.innerText.match(/step (\d+)\/\d+/)?.[1])) || -1);
  check("Network Alpha field present", await page.evaluate(() => [...document.querySelectorAll("label")].some(l => l.textContent.includes("Network Alpha"))));
  const previews = await page.evaluate(() => document.querySelectorAll(".gallery img, [data-testid='gallery'] img, button img").length);
  check("previews visible", previews > 0, previews);
  await tab("Loss");
  const marks = async () => page.evaluate(() => [...document.querySelectorAll("svg")].filter(s => s.getBoundingClientRect().height > 100)
    .map(s => s.querySelectorAll("path").length).reduce((a, b) => a + b, 0) + document.querySelectorAll("canvas").length);
  const s1 = await step(); const m1 = await marks(); await sleep(4000); const s2 = await step();
  check("Loss tab: page stays responsive, chart drawn, status keeps updating", (await alive()) && m1 > 0 && s2 > s1, `marks ${m1}, step ${s1} -> ${s2}`);
  await page.screenshot({ path: shot });
  await tab("Checkpoint Grid"); check("Checkpoint Grid tab responsive", await alive());
  await tab("Previews"); check("back to Previews responsive", await alive());
  await tab("Loss"); check("Loss again responsive", await alive());
  const box = (await page.$$(`xpath/.//label[contains(., "Trigger Word")]//textarea | .//label[contains(., "Trigger Word")]//input`))[0];
  await box.click({ clickCount: 3 }); await box.type("refresh_check"); await page.keyboard.press("Tab"); await sleep(2500);
  await page.reload({ waitUntil: "networkidle2" }); await sleep(3000);
  const val = await page.evaluate(() => [...document.querySelectorAll("label")].find(l => l.textContent.includes("Trigger Word"))?.querySelector("textarea,input")?.value);
  check("settings restored after reload", val === "refresh_check", JSON.stringify(val));
  await tab("Loss"); check("Loss tab responsive after reload", await alive());
  // --- model choice: each model has its own paths, memory options and preview settings
  const visible = async text => page.evaluate(t => [...document.querySelectorAll("label, button, span")]
    .some(e => e.textContent.includes(t) && e.getClientRects().length > 0 && e.offsetParent !== null), text);
  const pick = async name => { const [r] = await page.$$(`xpath/.//label[.//input[@type="radio"] and contains(normalize-space(.), "${name}")]`); await r.click(); await sleep(2500); };
  const shown = async () => ({ animaSwap: await visible("Block Swap (Low VRAM)"), animaPaths: await visible("Paths to Models <- Set Once"),
    mossPaths: await visible("Paths to MOSS-Image Models"), mossCaptions: await visible("Caption Source"),
    mossSwap: await visible("Text Encoder Block Swap"), mossSteps: await visible("Steps (MOSS-Image)"), mossNeg: await visible("Negative Prompt (MOSS-Image") });
  const onlyAnima = v => v.animaSwap && v.animaPaths && !v.mossPaths && !v.mossCaptions && !v.mossSwap && !v.mossSteps && !v.mossNeg;
  const onlyMoss = v => !v.animaSwap && !v.animaPaths && v.mossPaths && v.mossCaptions && v.mossSwap && v.mossSteps && v.mossNeg;
  let v = await shown(); check("Anima is the model at first: its controls show, MOSS-Image's don't", onlyAnima(v), JSON.stringify(v));
  await pick("MOSS-Image");
  v = await shown(); check("choosing MOSS-Image shows its paths, caption source, memory options and preview fields, and hides Anima's", onlyMoss(v), JSON.stringify(v));
  check("page still responsive after switching model mid-run", (await alive()) && (await step()) > 0);
  await page.screenshot({ path: shot.replace(/\.png$/, "_moss.png") });
  await page.reload({ waitUntil: "networkidle2" }); await sleep(3000);
  v = await shown(); check("the model choice survives a reload, with its controls", onlyMoss(v), JSON.stringify(v));
  await pick("Anima");
  v = await shown(); check("back to Anima: its controls return", onlyAnima(v), JSON.stringify(v));
  check("no page errors (Svelte update loop)", pageErrors === 0, pageErrors);
} finally { await browser.close(); }
process.exit(failed ? 1 : 0);
