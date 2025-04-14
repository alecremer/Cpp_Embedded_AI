project(ai)

# file(GLOB LIB_SOURCES "src/*.cpp")
# file(GLOB LIB_SOURCES "../src/*.cpp")

set(LIB_SOURCES
    ../src/layer.cpp
    ../src/ai.cpp
)

# Cria a biblioteca com os arquivos de implementação
add_library(ai_lib ${LIB_SOURCES})


# adiciona ao caminho de busca de cabecalhos
target_include_directories(ai_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}) 

