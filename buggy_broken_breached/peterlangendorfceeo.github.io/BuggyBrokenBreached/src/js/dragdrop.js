const dragListbox = document.getElementById('move-listbox');

if (dragListbox) {
    document.addEventListener('click', (e) => { 
        if (!e.target.closest('#move-listbox') && !e.target.closest('#btn-remove') && !e.target.closest('.btn-grid')) { 
            document.querySelectorAll('.list-item.selected').forEach(el => el.classList.remove('selected')); 
            if (window.syncHackerButtons) window.syncHackerButtons(); 
        } 
    });

    dragListbox.addEventListener('click', (e) => { 
        if (e.target.classList.contains('list-item')) { 
            document.querySelectorAll('.list-item.selected').forEach(el => el.classList.remove('selected')); 
            e.target.classList.add('selected'); 
            if (window.syncHackerButtons) window.syncHackerButtons(); 
        } 
    });

    dragListbox.addEventListener('dragstart', (e) => { 
        if(e.target.classList.contains('list-item')) { 
            isDraggingMove = true; 
            previousOrder = Array.from(dragListbox.children).map(c => c.innerText);
            setTimeout(() => e.target.classList.add('dragging'), 0); 
        } 
    });

    dragListbox.addEventListener('dragend', (e) => { 
        if(e.target.classList.contains('list-item')) { 
            e.target.classList.remove('dragging'); 
            isDraggingMove = false;
            if (currentPhase === 2) {
                const currentOrder = Array.from(dragListbox.children).map(c => c.innerText);
                if (JSON.stringify(previousOrder) !== JSON.stringify(currentOrder)) { 
                    if(window.logHackerAction) window.logHackerAction(`MEM_SHIFT --ptr_realloc=SUCCESS`); 
                }
            }
        } 
    });

    dragListbox.addEventListener('dragover', (e) => {
        if (currentPhase === 2) return;
        e.preventDefault();
        const afterElement = getDragAfterElement(dragListbox, e.clientY);
        const currentDraggable = document.querySelector('.dragging');
        if (currentDraggable) { 
            if (afterElement == null) { 
                dragListbox.appendChild(currentDraggable); 
            } else { 
                dragListbox.insertBefore(currentDraggable, afterElement); 
            } 
        }
    });
}

function getDragAfterElement(container, y) {
    const draggableElements = [...container.querySelectorAll('.list-item:not(.dragging)')];
    return draggableElements.reduce((closest, child) => {
        const box = child.getBoundingClientRect(); 
        const offset = y - box.top - box.height / 2;
        if (offset < 0 && offset > closest.offset) { 
            return { offset: offset, element: child } 
        } else { 
            return closest 
        }
    }, { offset: Number.NEGATIVE_INFINITY }).element;
}