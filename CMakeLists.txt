# pico-faces（cpldcpu/pico-faces: RP2350 で動く顔画像生成 DiT）を M5Stack Tab5（ESP32-P4）へ移植したもの。
#
# 上流のエンジン（upstream/engine、C99・整数のみ）は無改変で components/pf_engine がコンパイルする。
# 板固有の部分（画面、タッチ、USB シリアル、2 コア並列）だけを main/ に書いた。
#
#   git submodule update --init            上流（モデルの blob と rf_cfg.h を含む）を取る
#   ./idf.sh build                         既定モデル m3_decD_deep_full（4.02 MB、高品質）
#   ./idf.sh -DPF_MODEL=m3_long_cfg build  高速モデル（2.57 MB）
#   ./idf.sh -p /dev/ttyACM0 flash monitor
#
# ⚠️ -D で渡した PF_MODEL は CMake のキャッシュに残る。切り替えるときは build/ を消す。
cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED PF_MODEL)
    set(PF_MODEL "m3_decD_deep_full")
endif()
set(PF_MODEL "${PF_MODEL}" CACHE INTERNAL "pico-faces のモデル名（upstream/checkpoints/<name>）")
set(PF_MODEL_BIN "${CMAKE_CURRENT_LIST_DIR}/upstream/checkpoints/${PF_MODEL}/model.bin")
if(NOT EXISTS "${PF_MODEL_BIN}")
    message(FATAL_ERROR
        "モデル blob が無い: ${PF_MODEL_BIN}\n"
        "  git submodule update --init を実行したか、-DPF_MODEL の綴りを確認すること")
endif()
set(PF_MODEL_BIN "${PF_MODEL_BIN}" CACHE INTERNAL "モデル blob の絶対パス")
file(SIZE "${PF_MODEL_BIN}" PF_MODEL_BIN_SIZE)
message(STATUS "pico-faces: model ${PF_MODEL} (${PF_MODEL_BIN_SIZE} B) → .rodata")

# main とその依存だけをビルドする（IDF が lwip / wpa_supplicant などまで含めて RAM を食うのを防ぐ）
set(COMPONENTS main)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(pico_faces_tab5)
