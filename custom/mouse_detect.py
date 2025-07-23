import hid

print("Enumerating HID devices...")
found = False

for d in hid.enumerate():
    if d['vendor_id'] == 0x256f and d['product_id'] == 0xc635:
        print(f"Found: {d['product_string']} ({d['path']})")
        found = True
        try:
            dev = hid.device()
            dev.open_path(d['path'])
            print("✅ Device opened successfully!")
            dev.close()
        except Exception as e:
            print("❌ Open failed:", e)

if not found:
    print("❌ No matching device found.")
