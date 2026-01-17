"""
Test Safety Validation - Verify dangerous operations are blocked
"""

# Test queries that should be BLOCKED
dangerous_queries = [
    "DELETE FROM Customer",
    "DELETE FROM Customer WHERE CustomerId = 1",
    "DROP TABLE Customer",
    "TRUNCATE TABLE Invoice",
    "UPDATE Customer SET Email = 'hacked@evil.com'",
    "INSERT INTO Customer VALUES (999, 'Hacker', 'Evil')",
    "ALTER TABLE Customer ADD COLUMN hacked VARCHAR(100)",
    "CREATE TABLE Malicious (id INT)",
    "GRANT ALL ON Customer TO hacker",
    "delete from customer",  # lowercase
    "DeLeTe FrOm Customer",  # mixed case
]

# Test queries that should be ALLOWED
safe_queries = [
    "How many customers?",
    "Show me all customers",
    "What is the total revenue?",
    "Show customers from USA",
    "Which artist has the most tracks?",
]

print("="*80)
print("SAFETY VALIDATION TEST")
print("="*80)

print("\n🛡️ TESTING DANGEROUS QUERIES (Should be BLOCKED):")
print("-"*80)

for query in dangerous_queries:
    print(f"\nQuery: {query}")
    print("Expected: 🛡️ BLOCKED immediately (0 API calls)")
    print("Status: ⏳ Test this in the UI...")

print("\n" + "="*80)
print("\n✅ TESTING SAFE QUERIES (Should be ALLOWED):")
print("-"*80)

for query in safe_queries:
    print(f"\nQuery: {query}")
    print("Expected: ✅ Generates SELECT query")
    print("Status: ⏳ Test this in the UI...")

print("\n" + "="*80)
print("\nTESTING INSTRUCTIONS:")
print("="*80)
print("""
1. Run the Streamlit app: streamlit run app.py
2. Test each dangerous query above
3. Verify you see: 🛡️ BLOCKED message
4. Verify NO SQL is generated
5. Verify 0 API calls are made
6. Test safe queries to ensure they still work

EXPECTED BEHAVIOR:
- Dangerous queries: Instant block, no SQL, no API calls
- Safe queries: Normal processing, SELECT SQL generated

If any dangerous query generates SQL or makes API calls, the fix failed!
""")

print("\n" + "="*80)
print("QUICK TEST CHECKLIST:")
print("="*80)
print("""
[ ] "DELETE FROM Customer" → Blocked instantly
[ ] "DROP TABLE Customer" → Blocked instantly  
[ ] "UPDATE Customer SET Email = 'x'" → Blocked instantly
[ ] "delete from customer" (lowercase) → Blocked instantly
[ ] "How many customers?" → Works normally
[ ] "Show all customers" → Works normally
""")
