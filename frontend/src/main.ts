import "./style.css";

import { fetchExamples } from "./api";
import { h } from "./dom";
import { mountQuery } from "./query";
import { startWall } from "./wall";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const boxes = $<HTMLInputElement>("show-boxes");
const zones = $<HTMLInputElement>("show-zones");
for (const toggle of [boxes, zones]) {
  toggle.addEventListener("change", () => window.dispatchEvent(new Event("vms:overlay")));
}

const status = $("status");
startWall($("wall"), {
  intervalMs: 1500,
  showBoxes: () => boxes.checked,
  showZones: () => zones.checked,
  onStatus: (text, ok) => {
    status.textContent = text;
    status.classList.toggle("bad", !ok);
  },
});

const run = mountQuery($<HTMLFormElement>("ask"), $<HTMLInputElement>("question"), $("result"));
const chips = $("examples");
fetchExamples()
  .then((questions) => {
    chips.replaceChildren(
      ...questions.map((q) => {
        const chip = h("button", { type: "button", class: "chip" }, q);
        chip.addEventListener("click", () => run(q));
        return chip;
      }),
    );
  })
  .catch(() => chips.replaceChildren());
