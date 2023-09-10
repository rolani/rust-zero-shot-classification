#  rust-mlops-template

### Hugging Face Lyrics Analysis using Zero Shot Classification with SQLite

```
$ cargo run --quiet -- classify
Classify lyrics.txt
rock: 0.06948944181203842
pop: 0.27735018730163574
hip hop: 0.034089818596839905
country: 0.7835917472839355
latin: 0.6906086802482605
```

Print the lyrics:
```bash
cargo run --quiet -- lyrics | less | head
```

```bash
Lyrics lyrics.txt
Uh-uh-uh-uh, uh-uh
Ella despidió a su amor
El partió en un barco en el muelle de San Blas
El juró que volvería
Y empapada en llanto, ella juró que esperaría
Miles de lunas pasaron
Y siempre ella estaba en el muelle, esperando
Muchas tardes se anidaron
Se anidaron en su pelo y en sus labios
```

#### Hugging Face GPU Translation CLI

Goal:  Translate a spanish song to english
* `cargo new translate` and cd into it
fully working GPU Hugging Face Translation CLI in Rust

run it:   `time cargo run -- translate --path lyrics.txt`

```rust
/*A library that uses Hugging Face to Translate Text
*/
use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
use std::fs::File;
use std::io::Read;

//build a function that reads a file and returns a string
pub fn read_file(path: String) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

//build a function that reads a file and returns an array of the lines of the file
pub fn read_file_array(path: String) -> anyhow::Result<Vec<String>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let array = contents.lines().map(|s| s.to_string()).collect();
    Ok(array)
}


//build a function that reads a file and translates it
pub fn translate_file(path: String) -> anyhow::Result<()> {
    let model = TranslationModelBuilder::new()
        .with_source_languages(vec![Language::Spanish])
        .with_target_languages(vec![Language::English])
        .create_model()?;
    let text = read_file_array(path)?;
    //pass in the text to the model
    let output = model.translate(&text, None, Language::English)?;
    for sentence in output {
        println!("{}", sentence);
    }
    Ok(())
}
```

### Rust AWS Lambda

cd into `rust-aws-lambda`

* [Rust AWS Lambda docs](https://docs.aws.amazon.com/sdk-for-rust/latest/dg/lambda.html)
* Install AWS VS Code plugin and configure it to use your AWS account.
* See GitHub repo here: https://github.com/awslabs/aws-lambda-rust-runtime#deployment


To deploy: `make deploy` which runs: `cargo lambda build --release`

* Test inside of AWS Lambda console
* Test locally with:

```bash
cargo lambda invoke --remote \
  --data-ascii '{"command": "hi"}' \
  --output-format json \
  rust-aws-lambda
```

Result:
```bash
cargo lambda invoke --remote \
                --data-ascii '{"command": "hi"}' \
                --output-format json \
                rust-aws-lambda
{
  "msg": "Command hi executed.",
  "req_id": "1f70aff9-dc65-47be-977b-4b81bf83e7a7"
}
```

### OpenAI

#### Switching to Rust API Example

* install Rust via Rustup: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
Use Rust API for OpenAI (3rd party):  https://github.com/deontologician/openai-api-rust

* Create new project: `cargo new openai` and cd into it
* `make format` then `make lint` then `cargo run`

Working Example:

```bash
(.venv) @noahgift ➜ /workspaces/assimilate-openai/openai (main) $ cargo run -- complete -t "The rain in spain"
    Finished dev [unoptimized + debuginfo] target(s) in 0.14s
     Running `target/debug/openai complete -t 'The rain in spain'`
Completing: The rain in spain
Loves gets you nowhere
The rain in spain
```

`lib.rs`
```rust
/*This uses Open AI to Complete Sentences */

//accets the prompt and returns the completion
pub async fn complete_prompt(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let api_token = std::env::var("OPENAI_API_KEY")?;
    let client = openai_api::Client::new(&api_token);
    let prompt = String::from(prompt);
    let result = client.complete_prompt(prompt.as_str()).await?;
    Ok(result.choices[0].text.clone())
}

```


### Mixing Python and Rust

#### Using Rust Module from Python

* [Pyo3](https://pyo3.rs/v0.18.0/)
Try the getting started guide:

```bash
# (replace string_sum with the desired package name)
$ mkdir string_sum
$ cd string_sum
$ python -m venv .env
$ source .env/bin/activate
$ pip install maturin
```

* Run `maturin init` and then run `maturin develop` or `make develop`
* `python`
* Run the following python code
```python
import string_sum
string_sum.sum_as_string(5, 20)
```
The output should look like this: `'25'`

#### Using Python from Rust

Follow guide here: [https://pyo3.rs/v0.18.0/](https://pyo3.rs/v0.18.0/)

* install `sudo apt-get install python3-dev`
* `cargo new pyrust` and `cd pyrust`
* tweak `Cargo.toml` and add `pyo3`
* add source code to `main.rs`
* `make run`

```bash
Hello vscode, I'm Python 3.9.2 (default, Feb 28 2021, 17:03:44) 
[GCC 10.2.1 20210110]
```

Q:  Does the target binary have Python included?
A:  Maybe.  It does appear to be able to run Python if you go to the `target`
`/workspaces/rust-mlops-template/pyrust/target/debug/pyrust`

Follow up question, can I bring this binary to a "blank" codespace with no Python and what happens!


### Containerized Rust Examples

* `cargo new tyrscontainer` and cd into `tyrscontainer`
* copy a `Makefile` and `Dockerfile` from `webdocker`


Note that the rust build system container which is ~1GB is NOT in the final container image which is only 98MB.
```Dockerfile
FROM rust:latest as builder
ENV APP tyrscontainer
WORKDIR /usr/src/$APP
COPY . .
RUN cargo install --path .
 
FROM debian:buster-slim
RUN apt-get update && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/cargo/bin/$APP /usr/local/bin/$APP
#export this actix web service to port 8080 and 0.0.0.0
EXPOSE 8080
CMD ["tyrscontainer"]
````


The final container is very small i.e. 94MB
```bash
strings               latest    974d998c9c63   9 seconds ago   94.8MB
```

The end result is that you can easily test this web service and push to a cloud vendor like AWS and AWS App Runner.





