import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as fs from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

describe("packaging", () => {
  it("build copies src/runtime.py into dist/", () => {
    // The Python runtime is spawned as a subprocess at runtime, so the
    // build must ship it alongside the compiled JS
    const runtimePath = join(root, "dist", "runtime.py");
    assert.ok(fs.existsSync(runtimePath), "dist/runtime.py missing — build did not copy it");
    assert.ok(fs.statSync(runtimePath).size > 0, "dist/runtime.py is empty");
  });

  it("bin entrypoint exists", () => {
    assert.ok(fs.existsSync(join(root, "bin", "rlm.mjs")), "bin/rlm.mjs missing");
  });

  it("compiled main entry exists", () => {
    assert.ok(fs.existsSync(join(root, "dist", "main.js")), "dist/main.js missing");
  });
});
