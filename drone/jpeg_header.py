"""Builds the JPEG header that Variant-B drones leave off each video frame.

The drone only sends entropy-coded scan data. The phone app prepends a fixed
header (standard IJG quantisation tables scaled to the frame's quality, the
standard Annex K Huffman tables, 3 components, 1x1 sampling) and appends FFD9.
This module regenerates those headers from the standard tables.
"""
import struct

_LUMA_Q = [16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,
           14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,
           49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99]
_CHROMA_Q = [17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,
             47,66,99,99,99,99,99,99] + [99] * 32
_ZIGZAG = [0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,
           28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,
           54,47,55,62,63]
_DC_L_BITS = [0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0]; _DC_VALS = list(range(12))
_DC_C_BITS = [0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
_AC_L_BITS = [0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d]
_AC_L_VALS = bytes.fromhex(
 "01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a"
 "25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a"
 "838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2"
 "d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa")
_AC_C_BITS = [0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77]
_AC_C_VALS = bytes.fromhex(
 "000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718"
 "191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a"
 "82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9ca"
 "d2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9fa")

def _scaled(table, quality):
    scale = 5000 // quality if quality < 50 else 200 - quality * 2
    return bytes(min(255, max(1, (v * scale + 50) // 100)) for v in (table[z] for z in _ZIGZAG))

def _dht(cls_id, bits, vals):
    body = bytes([cls_id]) + bytes(bits) + bytes(vals)
    return b"\xff\xc4" + struct.pack(">H", len(body) + 2) + body

def jpeg_header(quality=50, width=640, height=360):
    out = b"\xff\xd8"
    out += b"\xff\xdb\x00\x43\x00" + _scaled(_LUMA_Q, quality)
    out += b"\xff\xdb\x00\x43\x01" + _scaled(_CHROMA_Q, quality)
    out += b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", height, width)
    out += b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    out += _dht(0x00, _DC_L_BITS, _DC_VALS) + _dht(0x10, _AC_L_BITS, _AC_L_VALS)
    out += _dht(0x01, _DC_C_BITS, _DC_VALS) + _dht(0x11, _AC_C_BITS, _AC_C_VALS)
    out += b"\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    return out
