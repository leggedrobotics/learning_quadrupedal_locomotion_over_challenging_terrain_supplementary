//
// Created by jolee
//

#ifndef MESSAGE_HPP
#define MESSAGE_HPP

#include "message_stream.hpp"

#define MSG(msg, severity) { std::stringstream messagestream; \
                                messagestream<<msg; \
                                Msg().stream(__FILE__, __LINE__, messagestream, severity); }

#define INFO(msg) MSG(msg, SEVERITY_INFO)
#define WARN(msg) MSG(msg, SEVERITY_WARN)
#define FATAL(msg) MSG(msg, SEVERITY_FATAL)
#define RETURN(con, msg) MSG(msg, SEVERITY_INFO)return;

#define INFO_IF(con, msg) if(con) MSG(msg, SEVERITY_INFO)
#define WARN_IF(con, msg) if(con) MSG(msg, SEVERITY_WARN)
#define FATAL_IF(con, msg) if(con) MSG(msg, SEVERITY_FATAL)
#define ASSERT(con, msg) if(!(con)) MSG(msg, SEVERITY_FATAL)
#define RETURN_IF(con, msg) if(con) {MSG(msg, RSEVERITY_INFO)return;}


#ifdef DEBUG
  #define DINFO(msg) INFO(msg)
  #define DWARN(msg) WARN(msg)
  #define DFATAL(msg) FATAL(msg)

  #define DINFO_IF(con, msg) INFO_IF(con, msg)
  #define DWARN_IF(con, msg) WARN_IF(con, msg)
  #define DFATAL_IF(con, msg) FATAL_IF(con, msg)

  #define DASSERT(con, msg) ASSERT(con, msg)
  #define DRETURN_IF(con, msg) INFO_IF(con, msg) return;
  #define DISNAN(val) FATAL_IF(isnan(val), #val<<" is nan");
  #define DISNAN_MSG(val, msg) FATAL_IF(isnan(val), msg);
#else
  #define DINFO(msg)
  #define DWARN(msg)
  #define DFATAL(msg)

  #define DINFO_IF(con, msg)
  #define DWARN_IF(con, msg)
  #define DFATAL_IF(con, msg)

  #define DASSERT(con, msg)
  #define DRETURN_IF(con, msg)
  #define DISNAN_MSG(val, msg)
  #define DISNAN
#endif

#endif // COMMON__MESSAGE_HPP
