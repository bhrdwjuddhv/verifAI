const git = require('isomorphic-git');
const fs = require('fs');
const http = require('isomorphic-git/http/node');
const path = require('path');

const dir = path.resolve('.');
const url = 'https://github.com/harshita16120/verifai.git';

async function main() {
  try {
    const token = process.env.GITHUB_TOKEN || '';
    console.log('🚀 Initializing Git repository with isomorphic-git...');
    await git.init({ fs, dir });

    console.log('🔗 Setting remote origin URL...');
    await git.addRemote({ fs, dir, remote: 'origin', url, force: true });

    console.log('📂 Staging project files recursively...');

    // Explicit root files
    const rootFiles = [
      'package.json',
      'next.config.js',
      'tailwind.config.ts',
      'tsconfig.json',
      '.gitignore',
      'middleware.ts',
      'vercel.json',
      'postcss.config.js',
      'README.md',
    ];

    for (const file of rootFiles) {
      if (fs.existsSync(path.join(dir, file))) {
        await git.add({ fs, dir, filepath: file });
        console.log(`  + Staged root file: ${file}`);
      }
    }

    // Stage directories recursively
    let stagedCount = 0;
    async function stageDir(relPath) {
      if (!fs.existsSync(path.join(dir, relPath))) return;
      const entries = fs.readdirSync(path.join(dir, relPath), { withFileTypes: true });
      for (const entry of entries) {
        const entryRel = path.join(relPath, entry.name).replace(/\\/g, '/');
        if (entry.isDirectory()) {
          if (entry.name !== 'node_modules' && entry.name !== '.next' && entry.name !== '.git') {
            await stageDir(entryRel);
          }
        } else {
          await git.add({ fs, dir, filepath: entryRel });
          stagedCount++;
        }
      }
    }

    await stageDir('app');
    await stageDir('components');
    await stageDir('lib');
    await stageDir('data');
    await stageDir('public');
    await stageDir('scripts');
    await stageDir('.github');

    console.log(`✅ Staged total ${stagedCount} files in subdirectories.`);

    console.log('📝 Creating commit...');
    const sha = await git.commit({
      fs,
      dir,
      author: {
        name: 'Harshita',
        email: 'harshita@verifai.open',
      },
      message: 'feat: Complete VerifAI Admin Panel, Multi-Modal Audio Pipeline, CI/CD & Free AI Integrations',
    });
    console.log(`✅ Commit created with SHA: ${sha}`);

    // Push to demo branch
    console.log(`📤 Pushing to ${url} (branch: demo)...`);
    const demoPushResult = await git.push({
      fs,
      http,
      dir,
      remote: 'origin',
      ref: 'demo',
      force: true,
      onAuth: () => ({ username: token }),
    });
    console.log('🎉 Push to demo completed:', demoPushResult.ok);

  } catch (err) {
    console.error('⚠️ Git operation report:', err.message, err.stack);
  }
}

main();
