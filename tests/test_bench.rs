use onnxbench::{benchmark, cli::Cli, setup_log};

#[test]
fn test_benchmark() {
    let inn = Cli {
        model_path: ".assets/yolov5nu.onnx".to_string(),
        input_shape: "images:1,3,640,640".to_string(),
        device: "cpu".to_string(),
        loop_n: 20,
        log: "info".to_string(),
    };

    setup_log(&inn.log);

    let res = benchmark(
        &inn.model_path,
        inn.loop_n,
        inn.parse_input_shape().unwrap(),
    );
    assert_eq!(res, Ok(()));
}

#[test]
fn test_benchmark2() {
    let inn = Cli {
        model_path: ".assets/yolov5nu.onnx".to_string(),
        input_shape: "images:1,3,256,256;imag2es".to_string(),
        device: "cpu".to_string(),
        loop_n: 10,
        log: "info".to_string(),
    };

    setup_log(&inn.log);

    let res = benchmark(
        &inn.model_path,
        inn.loop_n,
        inn.parse_input_shape().unwrap(),
    );
    assert_eq!(res, Ok(()));
}

#[test]
fn test_table() {
    use tabled::assert::assert_table;
    use tabled::{Table, Tabled};

    #[derive(Tabled)]
    struct Language {
        name: &'static str,
        designed_by: &'static str,
        invented_year: usize,
    }

    let languages = vec![
        Language {
            name: "C",
            designed_by: "Dennis Ritchie",
            invented_year: 1972,
        },
        Language {
            name: "Go",
            designed_by: "Rob Pike",
            invented_year: 2009,
        },
        Language {
            name: "Rust",
            designed_by: "Graydon Hoare",
            invented_year: 2010,
        },
    ];

    let table = Table::new(languages);

    assert_table!(
        table,
        "+------+----------------+---------------+"
        "| name | designed_by    | invented_year |"
        "+------+----------------+---------------+"
        "| C    | Dennis Ritchie | 1972          |"
        "+------+----------------+---------------+"
        "| Go   | Rob Pike       | 2009          |"
        "+------+----------------+---------------+"
        "| Rust | Graydon Hoare  | 2010          |"
        "+------+----------------+---------------+"
    );
}
