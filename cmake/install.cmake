
install(TARGETS ${DEPLOY_MAIN_NAME} DESTINATION bin)
install(TARGETS ${DEPLOY_LIB_NAME}  DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/model.h DESTINATION include/helmet_detection)

