import readline from 'readline';
import { createUser } from '../lib/admin/users';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function question(query: string): Promise<string> {
  return new Promise((resolve) => rl.question(query, resolve));
}

async function main() {
  console.log('=== VerifAI Admin Account Bootstrap ===\n');

  try {
    const email = await question('Enter Owner Email: ');
    if (!email || !email.includes('@')) {
      console.error('Invalid email address.');
      process.exit(1);
    }

    const password = await question('Enter Owner Password (min 8 chars): ');
    if (!password || password.length < 8) {
      console.error('Password must be at least 8 characters long.');
      process.exit(1);
    }

    const user = await createUser(email.trim(), password, 'owner');
    console.log('\n[SUCCESS] Owner account created successfully!');
    console.log(`ID: ${user.id}`);
    console.log(`Email: ${user.email}`);
    console.log(`Role: ${user.role}`);
  } catch (err: any) {
    console.error(`\n[ERROR] Failed to create owner account: ${err?.message || err}`);
  } finally {
    rl.close();
  }
}

main();
