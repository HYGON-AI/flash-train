file(READ "${INPUT}" assembly)
string(REPLACE "${SOURCE_SYMBOL}" "${DESTINATION_SYMBOL}" assembly "${assembly}")
file(WRITE "${OUTPUT}" "${assembly}")
