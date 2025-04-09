use std::fs::OpenOptions;

use log::LevelFilter;
use simplelog::{format_description, CombinedLogger, ConfigBuilder, SharedLogger, TermLogger, WriteLogger};
use time::UtcOffset;
use chrono::Local;


pub fn log_init(debug: bool, noterminal: bool) {
    let log_level = if debug {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };
    let conf_log = ConfigBuilder::new()
        .set_time_format_custom(format_description!("[year]-[month]-[day] [hour]:[minute]:[second].[subsecond]"))
        .set_time_offset(UtcOffset::from_whole_seconds(Local::now().offset().local_minus_utc()).unwrap())
        .set_max_level(LevelFilter::Debug)
        .set_location_level(LevelFilter::Debug)
        .build();

    let logger_file = WriteLogger::new(
        log_level,
        conf_log.clone(),
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(format!("wnum.{}.log", Local::now().format("%Y%m"))).unwrap()
    );
    let logger_vec: Vec<Box<dyn SharedLogger>> = if !noterminal {
        let logger = TermLogger::new(
            log_level,
            conf_log,
            simplelog::TerminalMode::Mixed,
            simplelog::ColorChoice::Auto
        );
        vec![logger, logger_file]
    } else {
        vec![logger_file]
    };

    CombinedLogger::init(logger_vec).unwrap();
}




