// Drives the real UI mid-run: tab switching must not freeze the page (Gradio's Svelte update loop), the loss chart
// draws and keeps updating, previews show, Network Alpha exists, settings come back after a reload.
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
  check("no page errors (Svelte update loop)", pageErrors === 0, pageErrors);
} finally { await browser.close(); }
process.exit(failed ? 1 : 0);
