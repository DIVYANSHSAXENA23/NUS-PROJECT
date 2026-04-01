# SplitSmart - Expense Splitting App

SplitSmart is a frontend-only web application designed to help groups fairly divide expenses and settle up simply. Perfect for group trips, roommates, or any shared expense scenarios.

Built entirely with HTML, CSS, and Vanilla JavaScript.

## Features

- **Add Group Members:** Quickly add people who are sharing expenses.
- **Support Multiple Currencies:** Choose from INR, USD, EUR, GBP, and SGD.
- **Log Expenses:** Record who paid and how much, with detailed descriptions.
- **Smart Settlement Algorithm:** Instantly calculates the most efficient way to settle all debts (who owes whom), minimizing the number of transactions.
- **Dark Mode:** A sleek dark theme that persists across pages.
- **State Persistence:** Data is reliably stored locally so you don't lose anything when moving between pages.
- **Responsive Design:** Works smoothly, with beautiful UI elements built from scratch without external libraries.

## Technologies Used

This project relies purely on core web technologies:
- **HTML5:** Semantic structuring across a clean 3-page flow.
- **CSS3:** Custom styling, responsive layouts, CSS variables, and animations (No Tailwind or Bootstrap).
- **Vanilla JavaScript:** DOM manipulation, greedy settlement algorithm, and state management (No React or external frameworks). 
- **Storage API:** `sessionStorage` for trip data, falling back to in-memory storage if blocked. `localStorage` for theme preference.

## How to Run Locally

You can run this project simply by opening it in your browser, or using a local server for optimal performance.

### Method 1: Python Local Server (Recommended)
1. Open your terminal or PowerShell.
2. Navigate to the project directory:
   ```bash
   cd path/to/project
   ```
3. Start the server:
   ```bash
   python -m http.server 8000
   ```
4. Open your web browser and go to: `http://localhost:8000`

### Method 2: Direct Open (Simplest)
Just double-click the `index.html` file in your file explorer to open it directly in your default browser.

## Project Structure

- `index.html` - Step 1: Manage group members and choose currency.
- `expenses.html` - Step 2: Log all expenses/receipts.
- `settlement.html` - Step 3: View total spending and the final "Who Pays Whom" breakdown.
- `app.js` - Contains core utilities, storage helpers, dark mode logic, and currency formatting.
- `style.css` - Custom styling rules and theme variables across the site.

## Course Project Note

This project was built to align with a Web Programming syllabus focusing on core concepts:
- **Module 2:** HTML tags, structure, and forms.
- **Module 3:** CSS styling, ID/Class selectors, box model.
- **Module 4 & 5:** JavaScript basics, DOM manipulation, event listeners, and control flow.

*Divide fairly. Settle simply.*
