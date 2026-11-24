from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Arithmetic Server")


@mcp.tool()
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation.

    Supported operations: add, sub, mul, div.
    """
    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero is not allowed"}
        result = first_num / second_num
    else:
        return {"error": f"Unsupported operation '{operation}'"}

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")