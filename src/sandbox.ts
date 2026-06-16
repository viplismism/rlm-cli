/**
 * OS-level sandbox for the model-generated Python subprocess.
 *
 * The RLM runs Python that the model writes — and a malicious or
 * prompt-injected context document can make that code do anything the user
 * can: read ~/.rlm/credentials and exfiltrate the API keys over the network.
 *
 * The child only ever talks to the parent over stdio pipes (LLM calls are
 * proxied through the parent), so it needs no network and no access to the
 * credential store. We wrap the spawn in a kernel-enforced sandbox that:
 *   - denies all network access      (kills exfiltration), and
 *   - hides ~/.rlm                    (credentials never enter the data flow).
 *
 * Mechanisms: macOS `sandbox-exec` (Seatbelt) and Linux `bwrap` (bubblewrap).
 * Each candidate is *probed* once — we only use it if it actually launches a
 * trivial Python in this environment — so a missing or unusable sandbox falls
 * back to running unsandboxed (with a warning) rather than breaking the tool.
 *
 * Opt out with RLM_NO_SANDBOX=1 (e.g. when you must let model code reach the
 * network or local files and you trust the input).
 */

import { spawnSync } from "node:child_process";
import * as path from "node:path";

export type SandboxMechanism = "seatbelt" | "bubblewrap" | "none";

export interface SandboxPlan {
	/** Executable to spawn (the sandbox wrapper, or python itself if none). */
	command: string;
	/** Arguments to the executable (sandbox flags + python + script args). */
	args: string[];
	/** Whether a kernel-enforced sandbox is actually in effect. */
	active: boolean;
	mechanism: SandboxMechanism;
	/** Human-readable reason when no sandbox is active. */
	reason?: string;
}

/** Cache probe results so restarts don't re-probe (keyed by mechanism). */
const probeCache = new Map<SandboxMechanism, boolean>();

function isDisabled(): boolean {
	const v = process.env.RLM_NO_SANDBOX;
	return v === "1" || v === "true" || v === "yes";
}

/**
 * Build a macOS Seatbelt profile. `(allow default)` keeps Python able to start
 * (it reads many system libraries); we then carve out network and the
 * credential store. Paths are embedded as quoted string literals.
 */
function seatbeltProfile(homeDir: string): string {
	const rlmDir = path.join(homeDir, ".rlm");
	// Seatbelt string literals are double-quoted; backslash/quote must be escaped.
	const esc = (p: string) => p.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
	return [
		"(version 1)",
		"(allow default)",
		"(deny network*)",
		`(deny file-read* (subpath "${esc(rlmDir)}"))`,
		`(deny file-write* (subpath "${esc(rlmDir)}"))`,
	].join("\n");
}

/** Run a candidate wrapper around a trivial python and report whether it works. */
function probe(mechanism: SandboxMechanism, command: string, args: string[], pythonCmd: string): boolean {
	const cached = probeCache.get(mechanism);
	if (cached !== undefined) return cached;
	let ok = false;
	try {
		const res = spawnSync(command, [...args, pythonCmd, "-c", "pass"], {
			stdio: "ignore",
			timeout: 5000,
		});
		ok = res.status === 0 && !res.error;
	} catch {
		ok = false;
	}
	probeCache.set(mechanism, ok);
	return ok;
}

/**
 * Plan how to spawn the Python runtime, sandboxed when possible.
 *
 * @param pythonCmd  the python executable ("python3"/"python")
 * @param scriptArgs args passed to python (e.g. [runtimePath])
 * @param homeDir    the user's home directory
 */
export function planPythonSpawn(pythonCmd: string, scriptArgs: string[], homeDir: string): SandboxPlan {
	const passthrough = (reason: string): SandboxPlan => ({
		command: pythonCmd,
		args: scriptArgs,
		active: false,
		mechanism: "none",
		reason,
	});

	if (isDisabled()) {
		return passthrough("disabled via RLM_NO_SANDBOX");
	}

	if (process.platform === "darwin") {
		const profile = seatbeltProfile(homeDir);
		const wrapperArgs = ["-p", profile];
		if (probe("seatbelt", "sandbox-exec", wrapperArgs, pythonCmd)) {
			return {
				command: "sandbox-exec",
				args: [...wrapperArgs, pythonCmd, ...scriptArgs],
				active: true,
				mechanism: "seatbelt",
			};
		}
		return passthrough("sandbox-exec unavailable or unusable");
	}

	if (process.platform === "linux") {
		const rlmDir = path.join(homeDir, ".rlm");
		// --bind / / keeps a normal read-write filesystem so Python behaves
		// normally; --unshare-net removes all network; --tmpfs over ~/.rlm
		// masks the credential store. --die-with-parent avoids orphans.
		const wrapperArgs = [
			"--die-with-parent",
			"--unshare-net",
			"--bind",
			"/",
			"/",
			"--dev",
			"/dev",
			"--proc",
			"/proc",
			"--tmpfs",
			rlmDir,
		];
		if (probe("bubblewrap", "bwrap", wrapperArgs, pythonCmd)) {
			return {
				command: "bwrap",
				args: [...wrapperArgs, pythonCmd, ...scriptArgs],
				active: true,
				mechanism: "bubblewrap",
			};
		}
		return passthrough("bubblewrap (bwrap) not installed or unusable");
	}

	return passthrough(`no sandbox mechanism available on ${process.platform}`);
}
