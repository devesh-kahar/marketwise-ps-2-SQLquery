# db_query_mysql.py
import mysql.connector
from tabulate import tabulate

class MySQLQuery:
    def __init__(self, host, user, password, database):
        """Initialize MySQL connection parameters"""
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        
    def connect(self):
        """Connect to MySQL database"""
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print(f"✓ Connected to MySQL database: {self.database}\n")
        except mysql.connector.Error as e:
            print(f"✗ Error connecting to database: {e}")
            
    def execute_query(self, query):
        """Execute a SQL query and display results"""
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(query)
            
            # Fetch all results
            results = cursor.fetchall()
            
            if results:
                print(f"\n✓ Query returned {len(results)} row(s)\n")
                print(tabulate(results, headers="keys", tablefmt="grid"))
            else:
                print("\n✓ Query executed successfully (0 rows returned)")
                
            cursor.close()
            return results
            
        except mysql.connector.Error as e:
            print(f"\n✗ SQL Error: {e}")
            return None
            
    def show_tables(self):
        """Show all tables in database"""
        query = "SHOW TABLES;"
        print("\n=== Database Tables ===")
        self.execute_query(query)
        
    def describe_table(self, table_name):
        """Show schema of a specific table"""
        query = f"DESCRIBE {table_name};"
        print(f"\n=== Schema for table: {table_name} ===")
        self.execute_query(query)
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("\n✓ Database connection closed")


def main():
    # MySQL connection details
    HOST = 'localhost'        # or '127.0.0.1'
    USER = 'THOR'             # your MySQL username
    PASSWORD = 'Thor_123'  # your MySQL password
    DATABASE = 'chinook'      # database name from your SQL file
    
    # Initialize database
    db = MySQLQuery(HOST, USER, PASSWORD, DATABASE)
    db.connect()
    
    print("=" * 60)
    print("MySQL Database Query Tool")
    print("=" * 60)
    print("\nCommands:")
    print("  - Type SQL query to execute")
    print("  - 'tables' - Show all tables")
    print("  - 'describe <table_name>' - Show table schema")
    print("  - 'quit' or 'exit' - Exit program")
    print("\n" + "=" * 60 + "\n")
    
    # Main loop
    while True:
        try:
            user_input = input("SQL> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
                
            if user_input.lower() == 'tables':
                db.show_tables()
                
            elif user_input.lower().startswith('describe '):
                table_name = user_input.split()[1]
                db.describe_table(table_name)
                
            else:
                db.execute_query(user_input)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    db.close()


if __name__ == "__main__":
    main()