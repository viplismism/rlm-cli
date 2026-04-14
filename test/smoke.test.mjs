import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const exec = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));
const bin = join(__dirname, "..", "bin", "rlm.mjs");

describe("rlm-cli smoke tests", () => {
  it("--help exits 0 and prints usage", async () => {
    const { stdout } = await exec("node", [bin, "--help"]);
    assert.match(stdout, /usage|rlm/i);
  });

  it("--version prints version string", async () => {
    const { stdout } = await exec("node", [bin, "--version"]);
    assert.match(stdout, /\d+\.\d+\.\d+/);
  });
});
