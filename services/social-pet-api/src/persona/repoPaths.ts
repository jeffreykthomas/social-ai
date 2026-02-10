import { existsSync } from 'node:fs';
import path from 'node:path';

function isRepoRoot(dir: string): boolean {
  // This repo has a stable top-level marker used by agents and tooling.
  return existsSync(path.join(dir, 'AGENTS.md')) && existsSync(path.join(dir, 'package.json'));
}

export function findRepoRoot(startDir?: string): string {
  let dir = startDir ?? process.cwd();
  for (let i = 0; i < 10; i += 1) {
    if (isRepoRoot(dir)) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return startDir ?? process.cwd();
}

export function resolveRepoPath(p: string, opts?: { startDir?: string }): string {
  if (path.isAbsolute(p)) return p;
  const root = findRepoRoot(opts?.startDir);
  return path.resolve(root, p);
}

