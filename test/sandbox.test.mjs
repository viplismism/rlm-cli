import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import * as os from "node:os";
import * as fs from "node:fs";
import * as path from "node:path";
import { planPythonSpawn } from "../dist/sandbox.js";

const PY = process.platform === "win32" ? "python" : "python3";
const HOME = os.homedir();

function runPlanned(plan, extraPyArgs) {
	// plan.args ends with the script args we passed in; here we pass python
	// inline code as the "script", so plan already wraps `python -c <code>`.
	return spawnSync(plan.command, plan.args, { encoding: "utf8", timeout: 15000 });
}

test("RLM_NO_SANDBOX disables sandboxing", () => {
	const prev = process.env.RLM_NO_SANDBOX;
	process.env.RLM_NO_SANDBOX = "1";
	try {
		const plan = planPythonSpawn(PY, ["runtime.py"], HOME);
		assert.equal(plan.active, false);
		assert.equal(plan.mechanism, "none");
		assert.equal(plan.command, PY);
		assert.deepEqual(plan.args, ["runtime.py"]);
	} finally {
		if (prev === undefined) delete process.env.RLM_NO_SANDBOX;
		else process.env.RLM_NO_SANDBOX = prev;
	}
});

test("a sandbox plan never loses the script arguments", () => {
	const plan = planPythonSpawn(PY, ["runtime.py", "--flag"], HOME);
	// Whether or not a sandbox is active, the final args must still run our script.
	assert.ok(plan.args.includes("runtime.py"));
	assert.ok(plan.args.includes("--flag"));
});

// The strong, platform-specific checks: when a sandbox is actually active,
// prove it blocks the two things the threat model cares about — network and
// credential reads — while leaving ordinary execution working.
const darwin = process.platform === "darwin";

test("macOS: sandbox is active and blocks network + ~/.rlm, allows normal code", { skip: !darwin }, () => {
	const plan = planPythonSpawn(PY, [], HOME);
	assert.equal(plan.active, true, `expected active sandbox, got: ${plan.reason}`);
	assert.equal(plan.mechanism, "seatbelt");
	assert.equal(plan.command, "sandbox-exec");

	const run = (code) => spawnSync(plan.command, [...plan.args, "-c", code], { encoding: "utf8", timeout: 15000 });

	// normal compute works
	const ok = run("print(6*7)");
	assert.equal(ok.status, 0);
	assert.equal(ok.stdout.trim(), "42");

	// outbound network is denied
	const net = run(
		"import socket\n" +
			"try:\n" +
			"    socket.create_connection(('1.1.1.1', 443), timeout=3); print('ALLOWED')\n" +
			"except Exception as e:\n" +
			"    print('BLOCKED', type(e).__name__)\n",
	);
	assert.equal(net.status, 0);
	assert.match(net.stdout, /BLOCKED/, `network was not blocked: ${net.stdout}`);

	// reading the credential store is denied
	const rlmDir = path.join(HOME, ".rlm");
	fs.mkdirSync(rlmDir, { recursive: true });
	const probe = path.join(rlmDir, "credentials.sandboxtest");
	fs.writeFileSync(probe, "SECRET=sk-should-not-leak");
	try {
		const read = run(
			`try:\n    print('LEAK', open(${JSON.stringify(probe)}).read())\nexcept Exception as e:\n    print('DENIED', type(e).__name__)\n`,
		);
		assert.equal(read.status, 0);
		assert.match(read.stdout, /DENIED/, `credential read was not blocked: ${read.stdout}`);
		assert.doesNotMatch(read.stdout, /sk-should-not-leak/);
	} finally {
		fs.rmSync(probe, { force: true });
	}
});
