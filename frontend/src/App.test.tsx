import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { HealthView } from "./App";
import type { Health } from "./api";

const ALL = { yunet: true, fer: true, vad: true, ser: true };

function render(health: Health, stale = false): string {
  return renderToStaticMarkup(<HealthView health={health} stale={stale} />);
}

describe("HealthView", () => {
  it("shows a ready CPU backend", () => {
    const html = render({ status: "ok", device: "cpu", models: ALL });
    expect(html).toContain("CPU");
    expect(html).toContain("всё готово");
    expect(html).not.toContain("Отсутствуют модели");
  });

  it("shows the GPU name", () => {
    const html = render({ status: "ok", device: "cuda", models: ALL, gpu_name: "NVIDIA GeForce RTX 3050 Laptop GPU" });
    expect(html).toContain("GPU (CUDA)");
    expect(html).toContain("RTX 3050");
  });

  it("explains the CUDA fallback in Russian", () => {
    const html = render({ status: "ok", device: "cpu", models: ALL, warning: "cuda_unavailable" });
    expect(html).toContain("видеокарта недоступна");
    expect(html).toContain('role="alert"');
  });

  it("uses a generic Russian text for an unknown warning", () => {
    const html = render({ status: "ok", device: "cpu", models: ALL, warning: "something_new" });
    expect(html).toContain("Сервер сообщил о проблеме");
  });

  it("lists missing models and the degraded status", () => {
    const html = render({ status: "degraded", device: "cpu", models: { ...ALL, fer: false, ser: false } });
    expect(html).toContain("работа ограничена");
    expect(html).toContain("Отсутствуют модели: Эмоции по лицу (EmotiEffLib), Эмоции по голосу (GigaAM-Emo)");
    expect(html).toContain("файл отсутствует");
  });

  it("marks stale data when the server is unreachable", () => {
    expect(render({ status: "ok", device: "cpu", models: ALL }, true)).toContain("Связь с сервером потеряна");
  });
});
