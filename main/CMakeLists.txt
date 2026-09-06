# ファーム本体（表示・タッチ・USB シリアル）と 2 コア並列（pf_par.c）。
# M5Unified / M5GFX は idf_component.yml の managed component として自動で要件に入る。
idf_component_register(
    SRCS "main.cpp" "pf_par.c"
    INCLUDE_DIRS "."
    REQUIRES pf_engine esp_timer esp_driver_usb_serial_jtag
)

# モデル blob を .rodata に埋め込む（シンボル _binary_model_bin_start / _end）。
# パスはトップの CMakeLists.txt が PF_MODEL から決める。
target_add_binary_data(${COMPONENT_LIB} "${PF_MODEL_BIN}" BINARY)
