import { authenticateUser } from '../lib/admin/auth';
import { isLockedOut } from '../lib/admin/lockout';
import { validateFileContent } from '../lib/admin/file-validation';
import { createUser } from '../lib/admin/users';

function toArrayBuffer(buf: Buffer): ArrayBuffer {
  const ab = new ArrayBuffer(buf.length);
  const view = new Uint8Array(ab);
  for (let i = 0; i < buf.length; ++i) {
    view[i] = buf[i];
  }
  return ab;
}

async function runVerificationChecks() {
  console.log('====================================================');
  console.log(' VERIFAI SECURITY HARDENING VERIFICATION SUITE ');
  console.log('====================================================\n');

  // Check 1: Account Lockout Test (6+ wrong attempts)
  console.log('--- TEST 1: Account Lockout Trigger (6+ failed logins) ---');
  const testEmail = 'attacker@test.com';
  for (let i = 1; i <= 6; i++) {
    const res = await authenticateUser(testEmail, 'wrongpass', '127.0.0.1');
    console.log(` Attempt ${i}: success=${res.success}, error="${res.error}", locked=${res.lockout?.locked}`);
  }
  const lockoutState = isLockedOut(testEmail);
  console.log(` Final Lockout Status: locked=${lockoutState.locked}, retryAfter=${lockoutState.retryAfter}\n`);

  // Check 3: Magic Byte Upload Validation
  console.log('--- TEST 3: Magic Byte File Content Validation ---');
  // Fake text buffer pretending to be a JPG
  const fakeJpgText = Buffer.from('THIS_IS_PLAIN_TEXT_NOT_A_JPEG_HEADER');
  const result1 = validateFileContent(toArrayBuffer(fakeJpgText), 'image');
  console.log(' Fake .jpg (Text content):', result1);

  // Genuine JPEG Header (FF D8 FF E0)
  const realJpgHeader = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46]);
  const result2 = validateFileContent(toArrayBuffer(realJpgHeader), 'image');
  console.log(' Genuine .jpg (JPEG Magic Bytes):', result2);

  // Check 5: User Store & Hashing Verification
  console.log('\n--- TEST 5: User Store & Hashing Verification ---');
  try {
    const safeUser = await createUser('testowner@verifai.dev', 'SecurePass123!', 'owner');
    console.log(' Created test owner account with bcrypt hashing:');
    console.log(' ', safeUser);
  } catch (err: any) {
    console.log(' Test owner account status:', err.message);
  }

  console.log('\nVerification suite script complete.');
}

runVerificationChecks().catch(console.error);
