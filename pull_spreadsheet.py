import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === Setup Credential & Connect ke Google Sheets ===
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

print(client)
# # === Buka Spreadsheet ===
sheet = client.open("Formulir Pra-Interview Karyawan SPPG (Jawaban)").worksheet("Form Responses 1")
# print(sheet)

# # === Ambil Semua Data ===
data = sheet.get_all_records()  # List of dict
print(data)

# # Contoh: Cari data berdasarkan nama
# def cari_data_karyawan(nama_dikenali):
#     for row in data:
#         if row['Nama'].lower() == nama_dikenali.lower():
#             return row
#     return None

# # Contoh Penggunaan
# matched_name = "Andi Saputra"  # Hasil dari face recognition
# karyawan_info = cari_data_karyawan(matched_name)

# if karyawan_info:
#     print(f"""
#     ✅ Karyawan dikenali:
#     - Nama       : {karyawan_info['Nama']}
#     - ID         : {karyawan_info['ID Karyawan']}
#     - Posisi     : {karyawan_info['Posisi']}
#     - Jam Masuk  : {karyawan_info['Jadwal Masuk']}
#     - Jam Keluar : {karyawan_info['Jadwal Keluar']}
#     """)
# else:
#     print("❌ Data karyawan tidak ditemukan di spreadsheet.")
