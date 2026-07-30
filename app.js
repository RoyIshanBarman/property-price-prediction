document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        searchBtn: document.getElementById('searchBtn'),
        locationInput: document.getElementById('locationInput'),
        resultsSection: document.getElementById('resultsSection'),
        resultsTable: document.getElementById('resultsTable'),
        exportBtn: document.getElementById('exportBtn'),
        errorDisplay: document.getElementById('errorDisplay'),
        propertyCount: document.getElementById('propertyCount'),
        loadingSpinner: document.getElementById('loadingSpinner')
    };

    let currentResults = [];

    async function handleSearch() {
        const location = elements.locationInput.value.trim();
        if (!location) {
            showError('Please enter a location');
            return;
        }

        resetUI();
        toggleLoading(true);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ Location: location })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || 'Server error occurred');
            }

            const data = await response.json();
            if (!data?.length) throw new Error('No properties found');

            currentResults = data;
            displayResults(data);
        } catch (error) {
            showError(error.message);
        } finally {
            toggleLoading(false);
        }
    }

    function displayResults(results) {
        if (elements.propertyCount) {
            elements.propertyCount.textContent = results.length;
        }

        if (elements.resultsTable) {
            elements.resultsTable.innerHTML = '';
        }

        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th class="text-left py-2 px-4">PROPERTY</th>
            <th class="text-right py-2 px-4">PRICE</th>
            <th class="text-right py-2 px-4">PREDICTED</th>
            <th class="text-right py-2 px-4">GROWTH</th>
            <th class="text-center py-2 px-4">RECOMMENDATION</th>
            <th class="text-center py-2 px-4">BEDROOMS</th>
            <th class="text-center py-2 px-4">BATHROOMS</th>
            <th class="text-right py-2 px-4">SIZE</th>
        `;
        elements.resultsTable.appendChild(headerRow);

        results.forEach(item => {
            const row = document.createElement('tr');
            row.className = 'border-t border-gray-200 hover:bg-gray-50';
            row.innerHTML = `
                <td class="py-2 px-4 font-medium">${escapeHTML(item.property_name)}</td>
                <td class="text-right py-2 px-4">${formatCurrency(item.actual_price)}</td>
                <td class="text-right py-2 px-4">${formatCurrency(item.predicted_price)}</td>
                <td class="text-right py-2 px-4 ${getGrowthColor(item.appreciation_rate)}">
                    ${item.appreciation_rate.toFixed(1)}%
                </td>
                <td class="text-center py-2 px-4">
                    <span class="px-2 py-1 rounded-full text-xs ${getRecommendationClass(item.recommendation)}">
                        ${item.recommendation}
                    </span>
                </td>
                <td class="text-center py-2 px-4">${item.bedrooms}</td>
                <td class="text-center py-2 px-4">${item.bathrooms}</td>
                <td class="text-right py-2 px-4">${item.size_sqft.toLocaleString()} sqft</td>
            `;
            elements.resultsTable.appendChild(row);
        });

        if (elements.resultsSection) {
            elements.resultsSection.classList.remove('hidden');
        }
    }

    //to CSV Function
    function exportToCSV() {
        if (!currentResults || currentResults.length === 0) {
            showError('No data available to export. Please perform a search first.');
            return;
        }

        try {
            const headers = [
                'Property_Name',
                'Actual_Price',
                'Predicted_Price',
                'Appreciation_Rate',
                'Recommendation',
                'Bedrooms',
                'Bathrooms',
                'Size_sqft'
            ];
            const rows = currentResults.map(item => [
                `"${item.property_name.replace(/"/g, '""')}"`,
                item.actual_price,
                item.predicted_price,
                item.appreciation_rate.toFixed(2),
                item.recommendation,
                item.bedrooms,
                item.bathrooms,
                item.size_sqft
            ]);

            //CSV content
            const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');

            //download
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            
            const today = new Date();
            const dateString = today.toISOString().split('T')[0];
            link.download = `property_recommendations_${dateString}.csv`;
            
            document.body.appendChild(link);
            link.click();
            
            setTimeout(() => {
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            }, 100);
        } catch (error) {
            console.error('Export failed:', error);
            showError('Failed to generate CSV file');
        }
    }

    function formatCurrency(amount) {
        return '$' + Math.round(amount).toLocaleString();
    }

    function escapeHTML(str) {
        return str.replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
    }

    function getGrowthColor(rate) {
        if (rate > 10) return 'text-green-600 font-bold';
        if (rate > 0) return 'text-blue-600';
        if (rate < 0) return 'text-red-600';
        return 'text-gray-600';
    }

    function getRecommendationClass(recommendation) {
        const classes = {
            "Strong Buy": "bg-green-100 text-green-800",
            "Buy": "bg-blue-100 text-blue-800",
            "Hold": "bg-yellow-100 text-yellow-800",
            "Don't Buy": "bg-red-100 text-red-800"
        };
        return classes[recommendation] || "bg-gray-100 text-gray-800";
    }

    function toggleLoading(isLoading) {
        if (elements.loadingSpinner) {
            elements.loadingSpinner.classList.toggle('hidden', !isLoading);
        }
        if (elements.searchBtn) {
            elements.searchBtn.disabled = isLoading;
            elements.searchBtn.innerHTML = isLoading 
                ? '<i class="fas fa-spinner fa-spin mr-2"></i> Analyzing...' 
                : '<i class="fas fa-search mr-2"></i> Search';
        }
    }

    function showError(message) {
        if (elements.errorDisplay) {
            elements.errorDisplay.textContent = message;
            elements.errorDisplay.classList.remove('hidden');
            setTimeout(() => {
                elements.errorDisplay.classList.add('hidden');
            }, 5000);
        }
    }

    function resetUI() {
        if (elements.errorDisplay) {
            elements.errorDisplay.textContent = '';
            elements.errorDisplay.classList.add('hidden');
        }
    }

    
    elements.searchBtn.addEventListener('click', handleSearch);
    elements.locationInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });
    
    
    elements.exportBtn.addEventListener('click', exportToCSV);
});