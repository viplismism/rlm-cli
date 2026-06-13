import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import * as fs from "node:fs";
import * as os from "node:os";

const exec = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));
const envModuleUrl = pathToFileURL(join(__dirname, "..", "dist", "env.js")).href;

// dist/env.js is a side-effect module that mutates process.env at import
// time, so each test imports it in a fresh subprocess with a controlled
// cwd, HOME, and environment, then inspects the resulting process.env.
async function runEnvModule({ dotenv, credentials, extraEnv = {} } = {}) {
  const cwd = fs.mkdtempSync(join(os.tmpdir(), "rlm-env-cwd-"));
  const home = fs.mkdtempSync(join(os.tmpdir(), "rlm-env-home-"));
  try {
    // Always write a cwd .env (possibly empty) so env.js never falls back
    // to the package root .env, which may exist on a dev machine
    fs.writeFileSync(join(cwd, ".env"), dotenv ?? "");
    if (credentials !== undefined) {
      fs.mkdirSync(join(home, ".rlm"));
      fs.writeFileSync(join(home, ".rlm", "credentials"), credentials);
    }
    const script = `await import(${JSON.stringify(envModuleUrl)}); process.stdout.write(JSON.stringify(process.env));`;
    const { stdout } = await exec(process.execPath, ["--input-type=module", "-e", script], {
      cwd,
      env: { PATH: process.env.PATH, HOME: home, ...extraEnv },
    });
    return JSON.parse(stdout);
  } finally {
    fs.rmSync(cwd, { recursive: true, force: true });
    fs.rmSync(home, { recursive: true, force: true });
  }
}

describe("env loader (dist/env.js)", () => {
  it("parses .env lines: export prefix, quotes, comments, malformed lines", async () => {
    const env = await runEnvModule({
      dotenv: [
        "# a comment",
        "",
        "PLAIN_KEY=plain-value",
        "export EXPORTED_KEY=exported-value",
        'DOUBLE_QUOTED="hello world"',
        "SINGLE_QUOTED='single value'",
        "EQUALS_IN_VALUE=a=b=c",
        "no-equals-sign-line",
        "  PADDED_KEY =  padded-value  ",
      ].join("\n"),
    });
    assert.equal(env.PLAIN_KEY, "plain-value");
    assert.equal(env.EXPORTED_KEY, "exported-value");
    assert.equal(env.DOUBLE_QUOTED, "hello world");
    assert.equal(env.SINGLE_QUOTED, "single value");
    assert.equal(env.EQUALS_IN_VALUE, "a=b=c");
    assert.equal(env.PADDED_KEY, "padded-value");
  });

  it("shell environment wins over .env", async () => {
    const env = await runEnvModule({
      dotenv: "SHARED_KEY=from-dotenv",
      extraEnv: { SHARED_KEY: "from-shell" },
    });
    assert.equal(env.SHARED_KEY, "from-shell");
  });

  it("loads ~/.rlm/credentials, with .env taking precedence", async () => {
    const env = await runEnvModule({
      dotenv: "BOTH_KEY=from-dotenv",
      credentials: "BOTH_KEY=from-credentials\nCRED_ONLY=cred-value",
    });
    assert.equal(env.BOTH_KEY, "from-dotenv");
    assert.equal(env.CRED_ONLY, "cred-value");
  });

  it("aliases GOOGLE_API_KEY to GEMINI_API_KEY and removes the original", async () => {
    const env = await runEnvModule({
      dotenv: "GOOGLE_API_KEY=google-key",
    });
    assert.equal(env.GEMINI_API_KEY, "google-key");
    assert.equal(env.GOOGLE_API_KEY, undefined);
  });

  it("does not overwrite an existing GEMINI_API_KEY when aliasing", async () => {
    const env = await runEnvModule({
      dotenv: "GOOGLE_API_KEY=google-key\nGEMINI_API_KEY=gemini-key",
    });
    assert.equal(env.GEMINI_API_KEY, "gemini-key");
    assert.equal(env.GOOGLE_API_KEY, undefined);
  });

  it("defaults RLM_MODEL when unset", async () => {
    const env = await runEnvModule({});
    assert.equal(env.RLM_MODEL, "claude-sonnet-4-6");
  });

  it("keeps an explicitly set RLM_MODEL", async () => {
    const env = await runEnvModule({
      extraEnv: { RLM_MODEL: "gpt-5.2" },
    });
    assert.equal(env.RLM_MODEL, "gpt-5.2");
  });
});
