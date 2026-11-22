// Render build/paper.html to build/paper.pdf with headless Chromium (Playwright).
// Used by build.sh when no LaTeX engine is installed. MathML renders natively.
const { chromium } = require("playwright");
const path = require("path");

(async () => {
  const html = path.resolve(__dirname, "build", "paper.html");
  const pdf = path.resolve(__dirname, "build", "paper.pdf");
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + html, { waitUntil: "networkidle" });
  await page.addStyleTag({
    content: `
      body { max-width: none; margin: 0; padding: 0; font-size: 10.5pt; line-height: 1.4; }
      img { max-width: 100%; page-break-inside: avoid; }
      figure, table, pre { page-break-inside: avoid; }
      h1, h2, h3 { page-break-after: avoid; }
      table { font-size: 9pt; }
      math[display="block"] { overflow-x: visible; }
    `,
  });
  await page.pdf({
    path: pdf,
    format: "A4",
    margin: { top: "22mm", bottom: "22mm", left: "20mm", right: "20mm" },
    displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate:
      '<div style="font-size:8pt;width:100%;text-align:center;color:#555">' +
      '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    printBackground: true,
  });
  await browser.close();
  console.log("wrote", pdf);
})();
