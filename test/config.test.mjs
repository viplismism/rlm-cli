import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { loadConfig } from "../dist/config.js";

const savedCwd = process.cwd();
const savedSubModel = process.env.RLM_SUB_MODEL;

// Run loadConfig() with cwd set to a temp dir containing the given yaml
// (loadConfig prefers rlm_config.yaml in cwd over the package root copy)
function loadWithYaml(yaml) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rlm-config-test-"));
  try {
    fs.writeFileSync(path.join(dir, "rlm_config.yaml"), yaml);
    process.chdir(dir);
    return loadConfig();
  } finally {
    process.chdir(savedCwd);
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

describe("loadConfig", () => {
  before(() => {
    delete process.env.RLM_SUB_MODEL;
  });

  after(() => {
    if (savedSubModel === undefined) {
      delete process.env.RLM_SUB_MODEL;
    } else {
      process.env.RLM_SUB_MODEL = savedSubModel;
    }
  });

  it("returns defaults for an empty config file", () => {
    const config = loadWithYaml("");
    assert.deepEqual(config, {
      max_iterations: 20,
      max_depth: 1,
      max_sub_queries: 50,
      truncate_len: 5000,
      metadata_preview_lines: 20,
      sub_model: "",
    });
  });

  it("parses values, skipping comments and blank lines", () => {
    const config = loadWithYaml(
      [
        "# RLM config",
        "",
        "max_iterations: 5",
        "truncate_len: 1000  # inline comment",
        "sub_model: gpt-4o-mini",
      ].join("\n"),
    );
    assert.equal(config.max_iterations, 5);
    assert.equal(config.truncate_len, 1000);
    assert.equal(config.sub_model, "gpt-4o-mini");
    // Untouched keys keep their defaults
    assert.equal(config.max_sub_queries, 50);
    assert.equal(config.metadata_preview_lines, 20);
  });

  it("strips quotes from string values", () => {
    const config = loadWithYaml('sub_model: "claude-haiku-4-5"');
    assert.equal(config.sub_model, "claude-haiku-4-5");
  });

  it("clamps numbers to their allowed ranges", () => {
    const config = loadWithYaml(
      [
        "max_iterations: 999",
        "max_sub_queries: 9999",
        "truncate_len: 100",
        "metadata_preview_lines: 1",
      ].join("\n"),
    );
    assert.equal(config.max_iterations, 100);
    assert.equal(config.max_sub_queries, 500);
    assert.equal(config.truncate_len, 500);
    assert.equal(config.metadata_preview_lines, 5);
  });

  it("rounds fractional numbers", () => {
    const config = loadWithYaml("max_iterations: 7.6");
    assert.equal(config.max_iterations, 8);
  });

  it("falls back to defaults for non-numeric values", () => {
    const config = loadWithYaml("max_iterations: lots\ntruncate_len: true");
    assert.equal(config.max_iterations, 20);
    assert.equal(config.truncate_len, 5000);
  });

  it("forces max_depth to 1 regardless of config", () => {
    const config = loadWithYaml("max_depth: 3");
    assert.equal(config.max_depth, 1);
  });

  it("uses RLM_SUB_MODEL env var when sub_model is absent", () => {
    process.env.RLM_SUB_MODEL = "gemini-2.5-flash";
    try {
      const config = loadWithYaml("max_iterations: 10");
      assert.equal(config.sub_model, "gemini-2.5-flash");
    } finally {
      delete process.env.RLM_SUB_MODEL;
    }
  });

  it("prefers sub_model from the config file over RLM_SUB_MODEL", () => {
    process.env.RLM_SUB_MODEL = "gemini-2.5-flash";
    try {
      const config = loadWithYaml("sub_model: gpt-4o-mini");
      assert.equal(config.sub_model, "gpt-4o-mini");
    } finally {
      delete process.env.RLM_SUB_MODEL;
    }
  });
});
