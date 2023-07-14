use actix_web::body::BoxBody;
use actix_web::{
    error, get,
    http::{header::ContentType, StatusCode},
    post, web, App, HttpResponse, HttpServer, Responder,
};
use discrete_logarithm_lib::{is_prime, shanks_algorithm_with_output, FpUnitsDiscLogSolver};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Deserialize)]
struct Inputs {
    prime: u32,
    base: u32,
    num: u32,
}

#[derive(Serialize)]
struct ShanksOutput {
    prime: u32,
    base: u32,
    num: u32,
    n: u32,
    log: u32,
    base_inverse: u32,
    collision_list1: HashMap<u32, i32>,
    collision_list2: HashMap<u32, i32>,
}

impl Responder for ShanksOutput {
    type Body = BoxBody;

    fn respond_to(self, req: &actix_web::HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}

#[derive(Debug)]
enum UserError {
    BadInput(String),
    InternalServer,
}

impl std::fmt::Display for UserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            UserError::BadInput(ref s) => write!(f, "{s}"),
            UserError::InternalServer => {
                write!(f, "An internal error occured. Please try again later.")
            }
        }
    }
}

impl error::ResponseError for UserError {
    fn error_response(&self) -> HttpResponse<BoxBody> {
        HttpResponse::build(self.status_code()).body(self.to_string())
    }

    fn status_code(&self) -> StatusCode {
        match *self {
            UserError::BadInput(_) => StatusCode::BAD_REQUEST,
            UserError::InternalServer => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

struct AppSolver {
    solver: Mutex<Option<FpUnitsDiscLogSolver>>,
}

impl AppSolver {
    fn new() -> Self {
        AppSolver {
            solver: Mutex::new(None),
        }
    }
}

#[post("/new/{prime}")]
async fn new_solver(
    path: web::Path<u32>,
    data: web::Data<AppSolver>,
) -> Result<HttpResponse, UserError> {
    let mut guard = if let Ok(guard_option) = data.as_ref().solver.lock() {
        guard_option
    } else {
        return Err(UserError::InternalServer);
    };
    // validate input from path
    let prime = path.into_inner();
    if !is_prime(prime) {
        return Err(UserError::BadInput(format!("{} is not prime", prime)));
    }
    *guard = Some(FpUnitsDiscLogSolver::new(prime));
    Ok(HttpResponse::Ok().body("created solver successfully\n"))
}

#[get("/modulus")]
async fn get_modulus(data: web::Data<AppSolver>) -> Result<HttpResponse, UserError> {
    let guard = if let Ok(guard_option) = data.as_ref().solver.lock() {
        guard_option
    } else {
        return Err(UserError::InternalServer);
    };

    if let Some(ref fp) = *guard {
        return Ok(HttpResponse::Ok().body(format!("prime modulus is {}\n", fp.prime)));
    }

    Ok(HttpResponse::Conflict().body("no solver created yet\n"))
}

#[get("/solve")]
async fn solve_discrete_logarithm_with_output(
    data: web::Json<Inputs>,
) -> Result<ShanksOutput, UserError> {
    // Get input from json
    let (prime, base, num) = (data.prime, data.base, data.num);
    // Check input is valid
    if !is_prime(prime) {
        return Err(UserError::BadInput(format!("{} is not prime", prime)));
    } else if base % prime == 0 {
        return Err(UserError::BadInput(format!(
            "{} is not a valid base, {} mod {} = 0",
            base, base, prime
        )));
    } else if num % prime == 0 {
        return Err(UserError::BadInput(format!(
            "{} has no logarithm mod {}, since {} mod {} = 0",
            num, prime, num, prime
        )));
    }

    // Get return data from solving discrete logarithm, if the log = 0 after matching we know no discrete logarithm exists
    let (log, base_inverse, n, list1, list2) = match shanks_algorithm_with_output(prime, base, num)
    {
        (Some(l), inv, ord, l1, l2) => (l, inv, ord, l1, l2),
        (None, inv, ord, l1, l2) => (0, inv, ord, l1, l2),
    };

    Ok(ShanksOutput {
        prime,
        base,
        num,
        log,
        base_inverse,
        n,
        collision_list1: list1,
        collision_list2: list2,
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let address = "127.0.0.1";
    let port = 8080;

    // Instantiate solver so it can be mutable between threads
    let app_solver = web::Data::new(AppSolver::new());

    HttpServer::new(move || {
        App::new()
            .app_data(app_solver.clone())
            .service(new_solver)
            .service(get_modulus)
            .service(solve_discrete_logarithm_with_output)
    })
    .bind((address, port))?
    .run()
    .await
}
