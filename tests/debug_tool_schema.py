"""Debug script to inspect tool schemas after Pydantic fix"""
import json
from src.tools.escalation_tools import (
    calculate_severity_score,
    generate_root_cause_analysis,
    create_audit_flag
)


def inspect_tool_schema(tool_func):
    """Attempt to inspect tool schema"""
    print(f"\n{'='*60}")
    print(f"Tool: {tool_func.__name__}")
    print('='*60)

    # Check various schema attributes
    schema_attrs = ['_schema', 'args_schema', 'schema', '__schema__', 'model_json_schema']

    found_schema = False
    for attr in schema_attrs:
        if hasattr(tool_func, attr):
            try:
                schema = getattr(tool_func, attr)
                if callable(schema):
                    schema = schema()

                print(f"\nFound schema in .{attr}:")
                print(json.dumps(schema, indent=2, default=str))
                found_schema = True

                # Check for additionalProperties in object-type parameters
                if isinstance(schema, dict) and 'properties' in schema:
                    for param_name, param_schema in schema.get('properties', {}).items():
                        if isinstance(param_schema, dict) and param_schema.get('type') == 'object':
                            has_additional_props = 'additionalProperties' in param_schema
                            status = "✅" if has_additional_props else "❌"
                            print(f"\n{status} Parameter '{param_name}' (object type):")
                            print(f"   additionalProperties: {param_schema.get('additionalProperties', 'MISSING')}")

                break
            except Exception as e:
                print(f"Error accessing .{attr}: {e}")

    if not found_schema:
        print("⚠️  No schema attribute found - may need alternate inspection method")
        print(f"   Available attributes: {[a for a in dir(tool_func) if not a.startswith('_')]}")


def main():
    print("\n" + "="*60)
    print("TOOL SCHEMA INSPECTION - Pydantic Fix Verification")
    print("="*60)

    # Inspect all fixed tools
    tools_to_inspect = [
        calculate_severity_score,
        generate_root_cause_analysis,
        create_audit_flag
    ]

    for tool in tools_to_inspect:
        inspect_tool_schema(tool)

    print("\n" + "="*60)
    print("Schema inspection complete!")
    print("="*60)
    print("\nLook for ✅ markers indicating 'additionalProperties: false' is present")
    print("If you see ❌ markers, the schema fix didn't work as expected.\n")


if __name__ == "__main__":
    main()
