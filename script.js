document.addEventListener('DOMContentLoaded', () => {
    const prevDayButton = document.getElementById('prev-day');
    const nextDayButton = document.getElementById('next-day');
    const currentDateSpan = document.getElementById('current-date');
    const curationContainer = document.getElementById('curation-container');

    const API_URL = 'https://mediatoday-bidaily-sokabogi-get-data-975969374955.asia-northeast3.run.app';
    const dataCache = {};

    // Helper function to get current time in KST (UTC+9)
    function getKSTDate(date = new Date()) {
        const now = date;
        const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
        const kstOffset = 9 * 60 * 60000;
        return new Date(utc + kstOffset);
    }

    let currentDate = getKSTDate();
    currentDate.setHours(0, 0, 0, 0);

    const today = getKSTDate();
    today.setHours(0, 0, 0, 0);

    const fiveDaysAgo = new Date(today);
    fiveDaysAgo.setDate(today.getDate() - 4);

    function formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}년 ${month}월 ${day}일`;
    }

    function showLoading() {
        curationContainer.innerHTML = '<p class="loading">데이터를 불러오는 중입니다...</p>';
    }

    async function fetchApiData(date, hour) {
        const dateStr = `${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, '0')}${String(date.getDate()).padStart(2, '0')}`;
        const hourStr = String(hour).padStart(2, '0');
        try {
            const response = await fetch(`${API_URL}?date=${dateStr}${hourStr}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const json = await response.json();
            return json.data || [];
        } catch (error) {
            console.error(`Error fetching data for ${dateStr}${hourStr}:`, error);
            return [];
        }
    }

    async function getAndRenderCuration(date) {
        showLoading();
        const dateString = date.toISOString().split('T')[0];

        if (dataCache[dateString]) {
            renderCuration(dataCache[dateString], date);
            return;
        }

        const [morningData, eveningData] = await Promise.all([
            fetchApiData(date, 7),
            fetchApiData(date, 17)
        ]);

        const combinedData = [...morningData, ...eveningData];
        const uniqueArticles = Array.from(new Map(combinedData.map(item => [item.id, item])).values());

        dataCache[dateString] = uniqueArticles;
        renderCuration(uniqueArticles, date);
    }

    function renderCuration(articles, date) {
        currentDateSpan.textContent = formatDate(date);
        curationContainer.innerHTML = '';

        if (!articles || articles.length === 0) {
            curationContainer.innerHTML = '<p>해당 날짜의 큐레이션 데이터가 없습니다.</p>';
            updateNavButtons();
            return;
        }

        let articlesToDisplay = articles;
        const kstNow = getKSTDate();
        // For today, only show morning articles if current KST time is before 6 PM (18:00)
        if (date.toDateString() === today.toDateString() && kstNow.getHours() < 18) {
            articlesToDisplay = articles.filter(a => a.id.includes('-07-'));
        }

        articlesToDisplay.forEach(item => {
            const card = document.createElement('div');
            card.className = 'card';

            const cleanTitle = item.title.replace(/^\[.*?\]\s*/, '');
            const cleanSubTitle = item.sub_title.replace(/^\[.*?\]\s*/, '');

            let content = item.content;
            const separatorIndex = content.indexOf('----------------');
            if (separatorIndex !== -1) {
                content = content.substring(0, separatorIndex);
            }

            card.innerHTML = `
                <h2>${cleanTitle}</h2>
                <h3>${cleanSubTitle}</h3>
                <p>${content}</p>
                <div class="read-more">
                    <a href="https://www.mediatoday.co.kr/news/articleList.html?sc_sdate=2024-10-28&sc_edate=2035-10-28&sc_area=A&view_type=sm&sc_word=AI+%EB%89%B4%EC%8A%A4+%EB%B8%8C%EB%A6%AC%ED%95%91" target="_blank">미디어오늘 웹페이지에서 읽어보기</a>
                </div>
            `;
            curationContainer.appendChild(card);
        });

        updateNavButtons();
    }

    function updateNavButtons() {
        prevDayButton.disabled = currentDate.toDateString() === fiveDaysAgo.toDateString();
        nextDayButton.disabled = currentDate.toDateString() === today.toDateString();
    }

    function changeDay(offset) {
        currentDate.setDate(currentDate.getDate() + offset);
        getAndRenderCuration(currentDate);
    }

    prevDayButton.addEventListener('click', () => changeDay(-1));
    nextDayButton.addEventListener('click', () => changeDay(1));

    getAndRenderCuration(currentDate);
});
