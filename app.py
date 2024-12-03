from fastapi import FastAPI

# Create an instance of FastAPI
app = FastAPI()

# Define a route for the homepage
@app.get("/")
def read_root():
    return {"Hello": "World"}

print(app)